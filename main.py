import json
import logging as log

import numpy as np
import matplotlib.pyplot as plt

import sympy_solve as sp_solve

# read json input and parse it
def read_input(filename: str) -> tuple[float, float, float, list[float]]:
    with open(filename) as f:
        input = json.load(f)
        return input['r'], input['v0'], input['s_max'], input['h']


def f(v0, s, r) -> float: return v0 * np.exp(s ** 2 / 2)


def rhs(v, s) -> float: return v * s


def rich(v_h2, v_h, p) -> np.ndarray:
        v_rich = np.zeros(len(v_h2))

        for i in range(len(v_h2)):
            v_rich[i] = (2 ** p * v_h[2 * i] - v_h2[i]) / (2 ** p - 1)
        
        return v_rich


def calc_err(x_analytics, x, method: str, h: float, logger) -> float:
    logger.debug('Starting calculate %s errors for h = %.4f', method, h)

    errs = []
    for i in range(0, len(x_analytics)):
        err = np.abs((x[i] - x_analytics[i]) / x_analytics[i])
        errs.append(err)
        logger.debug('%s_err[%i] = %.8f', method, i, err)

    max_err = np.max(np.array(errs))
    logger.warning('Error for h = %.4f and %s method: %.8f', h, method, max_err)

    return max_err

# class to avoid static and global values
class Solver():
    def __init__(self, filename: str) -> None:
        r, v0, s_max, h = read_input(filename)
        self.r = r
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

        self.rich_euler_errs = np.zeros(len(self.h) - 1)
        self.rich_adams_errs = np.zeros(len(self.h) - 1)

    # based on sympy solution
    def solve_analytics(self):
        self.v_analytics = [
            [f(self.v0, s, self.r) for s in s_val]
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
            if i != 0:
                self.rich_euler_errs[i - 1] = calc_err(
                    self.v_analytics[i - 1], self.euler_rich[i - 1],
                    'euler with richardson', self.h[i - 1], self.logger)

                self.rich_adams_errs[i - 1] = calc_err(
                    self.v_analytics[i - 1], self.adams_rich[i - 1],
                    'adams with richardson', self.h[i - 1], self.logger)

            self.euler_errs[i] = calc_err(
                self.v_analytics[i], self.euler_vals[i],
                'euler', self.h[i], self.logger)

            self.adams_errs[i] = calc_err(
                self.v_analytics[i], self.adams_vals[i],
                'adams', self.h[i], self.logger)

        euler_slope, _  = np.polyfit(np.log(self.h), np.log(self.euler_errs), deg=1)
        adams_slope, _ = np.polyfit(np.log(self.h), np.log(self.adams_errs), deg=1)
        euler_slope_rich, _ = np.polyfit(np.log(self.h[:-1]), np.log(self.rich_euler_errs), deg=1)
        adams_slope_rich, _ = np.polyfit(np.log(self.h[:-1]), np.log(self.rich_adams_errs), deg=1)
        self.logger.warning('Euler method accuracy: %.2f', euler_slope)
        self.logger.warning('Adams method accuracy: %.2f', adams_slope)
        self.logger.warning('Euler method with extrapolation accuracy: %.2f', euler_slope_rich)
        self.logger.warning('Adams method with extrapolation accuracy: %.2f', adams_slope_rich)

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

    def plot_errs(self, axs):
        log_h = np.log(self.h)
        x_err = np.linspace(-4, 0, 6)
        y_euler = np.array([x for x in x_err])
        y_adams = np.array([4 * x for x in x_err])

        axs.plot(x_err, y_euler, '--', label='Теор. точность метода Эйлера (точность = 1.0)')
        axs.plot(x_err, y_adams, '--', label='Теор. точность метода Адамса (точность = 4.0)')

        axs.plot(log_h, np.log(self.euler_errs), '-o', label='метод Эйлера')
        axs.plot(log_h, np.log(self.adams_errs), '-o', label='метод Адамса-Башфорта')

        axs.set_title('Зависимость логарифма ошибок от логарифма шага')
        axs.set_xlabel('log(h)')
        axs.set_ylabel('log(eps)')
        axs.legend()
        axs.grid(True)
                
    def run(self, axs):
        self.solve_analytics()

        for i in range(len(self.h)):
            self.solve_euler(i)
            self.solve_adams(i)
            if i != 0:
                self.euler_rich[i - 1] = rich(self.euler_vals[i - 1], self.euler_vals[i], 1)
                self.adams_rich[i - 1] = rich(self.adams_vals[i - 1], self.adams_vals[i], 4)

        self.calc_errors()
        self.plot_solve(axs[0])
        self.plot_errs(axs[1])

        plt.tight_layout()
        plt.show()

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

    # print solution from sympy
    logger.warning('Аналитическое решение через sympy: %s', sp_solve.solve())

    _, axs = plt.subplots(2, 1, figsize=(10, 10))

    solver = Solver('input.json')
    solver.set_logger(logger)
    solver.run(axs)


if __name__ == "__main__":
    main()
