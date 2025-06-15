import json
import logging as log
import os

import numpy as np
import matplotlib.pyplot as plt

# read json input and parse it
def read_input(filename: str) -> tuple[bool, float, float, list[float]]:
    with open(filename) as f:
        input = json.load(f)
        return input['debug'], input['v0'], input['s_max'], input['h']


def f(v0, s) -> float: return v0 * np.exp(s ** 2 / 2)


def rhs(v, s) -> float: return v * s


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



def calc_err(x_analytics, x, method: str, h: float, logger) -> float:
    logger.debug('Starting calculate %s errors for h = %.4f', method, h)

    errs = []
    for i in range(0, len(x_analytics)):
        err = np.abs((x[i] - x_analytics[i]) / x_analytics[i]) * 100
        errs.append(err)
        logger.debug('%s_err[%i] = %.8f', method, i, err)

    max_err = np.max(np.array(errs))
    logger.warning('Error for h = %.4f and %s method: %.8f', h, method, max_err)

    return max_err

# class to avoid static and global values
class Solver():
    def __init__(self, filename: str) -> None:
        debug, v0, s_max, h = read_input(filename)
        self.debug = debug
        self.v0 = v0
        self.h = h
        self.s_max = s_max

        # setting arrays with zeros 
        self.s_vals = [np.arange(0, s_max + h_val, h_val) for h_val in h]
        self.n = np.array([len(s_val) - 1 for s_val in self.s_vals])

        self.euler_vals = [np.zeros(self.n[i] + 1) for i in range(len(self.h))]
        self.adams_vals = [np.zeros(self.n[i] + 1) for i in range(len(self.h))]

        self.euler_rich = [np.zeros(self.n[i] + 1) for i in range(1, len(self.h))]
        self.adams_rich = [np.zeros(self.n[i] + 1) for i in range(1, len(self.h))]

        self.euler_errs = np.zeros(len(self.h))
        self.adams_errs = np.zeros(len(self.h))

    # based on sympy solution
    def solve_analytics(self):
        self.v_analytics = [
            [f(self.v0, s) for s in s_val]
            for s_val in self.s_vals
        ]

    def solve_euler(self, ind: int):
        v_vals = np.zeros(self.n[ind] + 1)
        v_vals[0] = self.v0

        for i in range(self.n[ind]):
            v_vals[i + 1] = v_vals[i] + self.h[ind] * rhs(v_vals[i], self.s_vals[ind][i]) 

        self.euler_vals[ind] = v_vals

    def solve_adams(self, ind):
        v_vals = np.zeros(self.n[ind] + 1)
        v_vals[0] = self.v0

        # euler method for the fist 4 points
        for i in range(3):
            v_vals[i + 1] = v_vals[i] + self.h[ind] * rhs(v_vals[i], self.s_vals[ind][i]) 

        # start adams method
        for i in range(3, self.n[ind]):
            v_vals[i + 1] = v_vals[i] + self.h[ind] / 24 * (
                55 * rhs(v_vals[i], self.s_vals[ind][i]) -
                59 * rhs(v_vals[i - 1], self.s_vals[ind][i - 1]) + 
                37 * rhs(v_vals[i - 2], self.s_vals[ind][i - 2]) -
                9 * rhs(v_vals[i - 3], self.s_vals[ind][i - 3])
            )

        self.adams_vals[ind] = v_vals
    
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
        y_vals = np.array([f(self.v0, x) for x in x_vals])
        
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

    def plot_errs(self, axs):
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
                
    def run(self, axs, fig):
        self.solve_analytics()

        for i in range(len(self.h)):
            self.solve_euler(i)
            self.solve_adams(i)

        self.calc_errors()
        self.plot_solve(axs[0])
        self.plot_errs(axs[1])

        plt.tight_layout()
        if self.debug:
            plt.show()
        else:
            if not os.path.exists('out'):
                os.mkdir('out')
            fig.savefig('out/result.png', dpi=100)

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

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    solver = Solver('input.json')
    solver.set_logger(logger)
    solver.run(axs, fig)


if __name__ == "__main__":
    main()
