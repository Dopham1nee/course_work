import json
import logging as log
import os

import numpy as np
import matplotlib.pyplot as plt

import sympy_solve as sp_solve

# read json input and parse it
def read_input(filename: str) -> tuple[bool, float, float, float, list[float]]:
    with open(filename) as f:
        input = json.load(f)
        return input['debug'], input['r'], input['v0'], input['s_max'], input['h']


def f(v0, s, r) -> float: return v0 * np.exp(-s / r)


def rhs(v, r) -> float: return -v / r


def richardson_accuracy(y_h, y_h2, y_h4):
    p_vals = []
    for i in range(len(y_h)):
        numerator = y_h[i] - y_h2[2 * i]
        denominator = y_h2[2 * i] - y_h4[4 * i]
        if np.abs(denominator) < 1e-12:
            continue
        else:
            p_vals.append(np.log2(np.abs(numerator / denominator)))
    return np.percentile(p_vals, 95)



def calc_err(y_analytics, y, method: str, h: float, logger) -> float:
    if method != 'none':
        logger.debug('Starting calculate %s errors for h = %.3f', method, h)

    errs = []
    for i in range(0, len(y_analytics)):
        err = np.abs((y[i] - y_analytics[i]) / y_analytics[i]) * 100
        if err == 100:
            print(f"{y_analytics[i]}, {y[i]}")
        errs.append(err)
        if method != 'none': logger.debug('%s_err[%i] = %.8f', method, i, err)

    max_err = np.max(np.array(errs))
    if method != 'none':
        logger.warning('Error for h = %.3f and %s method: %.8f', h, method, max_err)

    return max_err


def euler_round(x: np.ndarray, h, n, start, b):
    x[0] = start
    for i in range(n):
        x[i + 1] = x[i] + h * rhs(x[i], b)

    return x


def adams_round(y: np.ndarray, h, n, start, b):
    # first 4 points from euler
    y[:4] = euler_round(y[:4], h, 3, start, b)
    
    for i in range(3, n):
        y[i + 1] = y[i] + (h / 24) * (
            55 * rhs(y[i], b) -
            59 * rhs(y[i - 1], b) +
            37 * rhs(y[i - 2], b) -
            9 * rhs(y[i - 3], b)
        )
    
    return y


def find_opt_grid(
    h: float,
    err: float,
    start: float,
    y_max: float,
    b: float,
    method,
):
    while 0.01 > err or err > 0.1:
        if err > 0.1:
            h /= 2
        elif err < 0.01:
            h *= 2

        x_vals = np.arange(0, y_max + h, h)
        y_analytics = np.array([f(start, x, b) for x in x_vals])
        n = len(x_vals) - 1
        y = np.zeros(n + 1)

        y = method(y, h, len(x_vals) - 1, start, b)
        err = calc_err(y_analytics, y, 'none', h, None)
    
    return h, err


# class to avoid static and global values
class Solver():
    def __init__(self, filename: str) -> None:
        debug, r, v0, s_max, h = read_input(filename)
        self.debug = debug
        self.r = r
        self.v0 = v0
        self.h = h
        self.s_max = s_max

        # setting arrays with zeros 
        self.s_vals = [np.arange(0, s_max + h_val, h_val) for h_val in h]
        self.n = np.array([len(s_val) - 1 for s_val in self.s_vals])

        self.euler_vals = [np.zeros(self.n[i] + 1) for i in range(len(self.h))]
        self.adams_vals = [np.zeros(self.n[i] + 1) for i in range(len(self.h))]

        self.euler_errs = np.zeros(len(self.h))
        self.adams_errs = np.zeros(len(self.h))

    # based on sympy solution
    def solve_analytics(self):
        self.v_analytics = [
            [f(self.v0, s, self.r) for s in s_val]
            for s_val in self.s_vals
        ]

    def solve_euler(self):
        for i in range(len(self.h)):
            v_vals = np.zeros(self.n[i] + 1)
            v_vals = euler_round(v_vals, self.h[i], self.n[i], self.v0, self.r)
            self.euler_vals[i] = v_vals

    def solve_adams(self):
        for i in range(len(self.h)):
            v_vals = np.zeros(self.n[i] + 1)
            v_vals = adams_round(v_vals, self.h[i], self.n[i], self.v0, self.r)
            self.adams_vals[i] = v_vals
    
    def calc_errors(self):
        for i in range(len(self.n)):
            self.euler_errs[i] = calc_err(
                self.v_analytics[i], self.euler_vals[i],
                'euler', self.h[i], self.logger)

            self.adams_errs[i] = calc_err(
                self.v_analytics[i], self.adams_vals[i],
                'adams', self.h[i], self.logger)

        euler_slope, _  = np.polyfit(np.log(self.h), np.log(self.euler_errs), deg=1)
        adams_slope, _ = np.polyfit(np.log(self.h), np.log(self.adams_errs), deg=1)
        self.logger.warning('Euler method accuracy: %.2f', euler_slope)
        self.logger.warning('Adams method accuracy: %.2f', adams_slope)

        self.logger.warning(
            'Euler method with extrapolation accuracy: %.2f',
            richardson_accuracy(self.euler_vals[-3], self.euler_vals[-2], self.euler_vals[-1]))
        self.logger.warning(
            'Adams method with extrapolation accuracy: %.2f',
            richardson_accuracy(self.adams_vals[-3], self.adams_vals[-2], self.adams_vals[-1]))

    def plot_solve(self, axs):
        cnt = len(self.h) // 2
        x_vals = np.linspace(0, self.s_max, 3000)
        y_vals = np.array([f(self.v0, x, self.r) for x in x_vals])
        
        axs.plot(x_vals, y_vals, label='Аналитическое решение', linewidth=2)
        # Euler solutions
        axs.plot(self.s_vals[0], self.euler_vals[0], '--', label=f'Метод Эйлера h = {self.h[0]}')
        axs.plot(self.s_vals[cnt], self.euler_vals[cnt], '--', label=f'Метод Эйлера h = {self.h[cnt]}')
        axs.plot(self.s_vals[-1], self.euler_vals[-1], '--', label=f'Метод Эйлера h = {self.h[-1]}')
        # Adams solutions 
        axs.plot(self.s_vals[0], self.adams_vals[0], '--', label=f'Метод Адамса-Башфорта h = {self.h[0]}')
        axs.plot(self.s_vals[cnt], self.adams_vals[cnt], '--', label=f'Метод Адамса-Башфорта h = {self.h[cnt]}')
        axs.plot(self.s_vals[-1], self.adams_vals[-1], '--', label=f'Метод Адамса-Башфорта h = {self.h[-1]}')

        axs.set_title('Решение уравнения')
        axs.set_xlabel('S, м')
        axs.set_ylabel('v(S), м/с')
        axs.legend()
        axs.grid(True)
        pass

    def plot_convergence(self, axs):
        log_h = np.log(self.h)
        log_euler = np.log(self.euler_errs)
        log_adams = np.log(self.adams_errs)
        
        # calculating b from equal y = p * x + b
        # p - accuracy of the method
        b_euler = log_euler[0] - log_h[0]
        b_adams = log_adams[0] - 4 * log_h[0]

        y_euler = np.array([(x + b_euler) for x in log_h])
        y_adams = np.array([(4 * x + b_adams) for x in log_h])

        axs.plot(log_h, y_euler, '--', label='Теор. точность метода Эйлера (точность = 1.0)')
        axs.plot(log_h, y_adams, '--', label='Теор. точность метода Адамса (точность = 4.0)')

        axs.plot(log_h, np.log(self.euler_errs), '-o', label='метод Эйлера')
        axs.plot(log_h, np.log(self.adams_errs), '-o', label='метод Адамса-Башфорта')

        axs.set_title('Зависимость логарифма ошибок от логарифма шага')
        axs.set_xlabel('log(h)')
        axs.set_ylabel('log(eps)')
        axs.legend()
        axs.grid(True)

    def research_const(self, h, axs):
        b_vals = np.array([self.r / 2, self.r, self.r * 2])
        s_vals = np.arange(0, self.s_max + h, h)
        n = len(s_vals) - 1

        for b in b_vals:
            y = np.zeros(n + 1)
            y = adams_round(y, h, n, self.v0, b)
            axs.plot(s_vals, y, label=f'Решения для r = {b}')

        axs.set_title('Решения для разных констант')
        axs.set_xlabel('S, м')
        axs.set_ylabel('v(s), м/с')
        axs.legend()
        axs.grid(True)
                
    def run(self, axs, figs):
        self.solve_analytics()
        self.solve_euler()
        self.solve_adams()

        self.calc_errors()
        self.plot_solve(axs[0])
        self.plot_convergence(axs[1])

        h_euler, err_euler = find_opt_grid(self.h[0], self.euler_errs[0], self.v0, self.s_max, self.r, euler_round)
        h_adams, err_adams = find_opt_grid(self.h[0], self.adams_errs[0], self.v0, self.s_max, self.r, adams_round)

        self.logger.warning("Optimized step | euler: %.4f | adams: %.4f |", h_euler, h_adams)
        self.logger.warning("Error for step | euler: %.4f | adams: %.4f |", err_euler, err_adams)

        self.research_const(h_adams, axs[2])

        plt.tight_layout()

        if self.debug:
            plt.show()
        else:
            if not os.path.exists('out'):
                os.mkdir('out')
            figs[0].savefig('out/solve.png', dpi=100)
            figs[1].savefig('out/convergence.png', dpi=100)
            figs[2].savefig('out/b-research.png', dpi=100)


    def set_logger(self, logger): self.logger = logger


def main():
    # for logging errors 
    logger = log.getLogger('app')
    logger.setLevel(log.DEBUG)

    formatter = log.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = log.StreamHandler()
    console_handler.setLevel(log.WARNING)
    console_handler.setFormatter(formatter)

    file_handler = log.FileHandler("errs.txt", mode='w')
    file_handler.setLevel(log.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.warning("Sympy analytics solution: %s", sp_solve.solve())

    fig1, axs1 = plt.subplots()
    fig2, axs2 = plt.subplots()
    fig3, axs3 = plt.subplots()

    solver = Solver('input.json')
    solver.set_logger(logger)
    solver.run([axs1, axs2, axs3], [fig1, fig2, fig3])


if __name__ == "__main__":
    main()
