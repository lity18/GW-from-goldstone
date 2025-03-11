import numpy as np
import scipy as sci
from mpi4py import MPI
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import os 
import math as m
import scipy.signal


def before_pt_evolution_6(t0, y, k):
    y2 = - 3 * y[1] / t0 + (y[0] - y[0] ** 3 + k * y[0] ** 5)
    return np.array([y[1], y2])


def after_pt_evolution_6(t0, y, k):
    y2 = - 3 * y[1] / t0 - (y[0] - y[0] ** 3 + k * y[0] ** 5)
    return np.array([y[1], y2])


def bubble_6(k):
    f_x = 0
    t_x = ((1 + (1 - 4 * k) ** 0.5) / 2 / k) ** 0.5
    n2 = ((1 - (1 - 4 * k) ** 0.5) / 2 / k) ** 0.5
    n1 = (t_x - n2) * 0.9 + n2
    r = np.linspace(0.002, 25, 1000)
    for i in range(100):
        sol = solve_ivp(before_pt_evolution_6, [0.0001, 25], [n1, 0], method="Radau", args=[k], dense_output=True)
        min = np.min(sol.sol(r)[0])
        if min > f_x:
            n1 = (t_x + n1) / 2
        else:
            break
    for i in range(100):
        n = (n1 + n2) / 2
        sol = solve_ivp(before_pt_evolution_6, [0.0001, 25], [n, 0], method="Radau", args=[k], dense_output=True)
        min = np.min(sol.sol(r)[0])
        if min >= f_x:
            n2 = n
        else:
            n1 = n
        if n1 - n2 == 0:
            break
    sol_out = solve_ivp(before_pt_evolution_6, [0.0001, 25], [n, 0], method="Radau", args=[k], dense_output=True)
    sol_in = solve_ivp(after_pt_evolution_6, [0.0001, 1000], [n, 0], method="Radau", args=[k], dense_output=True)
    y1 = sol_out.sol(r)[0]
    r1 = n / 4 + f_x * 3 / 4
    r2 = n * 3 / 4 + f_x / 4
    left = np.array(np.where(y1 >= r2))[0, -1]
    right = np.array(np.where(y1 <= r1))[0, 0]
    delta = r[right] - r[left]
    r0 = 1.5 * delta + r[np.array(np.where(y1 <= r1))[0, 0]]
    r1 = 4 * delta + r[np.array(np.where(y1 <= r1))[0, 0]]
    return sol_in, sol_out, r1, r0, delta, n


def v_x_6(x, k):
    return x * x / 2 - x ** 4 / 4 + k * x ** 6 / 6
