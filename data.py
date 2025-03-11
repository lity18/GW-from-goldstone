import numpy as np
from mpi4py import MPI
import time
import os
import datetime
import matplotlib.pyplot as plt
import scipy.special as spec
import scipy as sci 
import function as f




def energy_ratio_data():
    kap = 0.1
    tv = ((1 + (1 - 4 * kap) ** 0.5) / 2 / kap) ** 0.5
    tv_1 = ((1 - (1 - 4 * kap) ** 0.5) / 2 / kap) ** 0.5
    fv = 0
    sol_in, sol_out, r01, r0, delta, phi0 = f.bubble_6(kap)

    gamma0 = np.array([5, 10, 15, 20])
    phase0 = np.array([1, 6, 9, 12])
    data = np.zeros((5, 5, 2))
    N = 10

    for jj in range(np.size(phase0)):
        data[jj, 0, 0] = phase0[jj]
        for ii in range(np.size(gamma0)):
            gamma = gamma0[ii]
            phase = phase0[jj]
            dx = delta / gamma / 50
            t0 = 0.9 * gamma * r0
            t_i = 0.0002
            d = gamma * r0

            dt = dx / 5
            Nt = int(d / dt * (N - 2))  

            N_slice = 500

            slice_rate = int(Nt / N_slice)
            s_0 = np.zeros(N_slice)
            t_0 = np.zeros(N_slice)

            for i in range(N_slice):
                t = i * slice_rate * dt + t0
                t_0[i] = t / d
                s_0[i] = 1 / 3 * d * np.pi * (t ** 2 - d ** 2) * (f.v_x_6(fv, kap) - f.v_x_6(tv, kap))
    
            names = "/home/storage0/users/tingyuli/storage4/field-evolution-code/field-evolution-code/two_bubble/different_boost/" + str(gamma) + "_phase_" + str(phase) + "over16pi"
            datas = np.load(names + "/energy.npy")
            num = -50

            data[jj, ii + 1, 0] = ((datas[:-1, 0]) / s_0)[num]
            data[jj, ii + 1, 1] = (datas[:-1, 1] / s_0)[num]
    np.save("/home/storage0/users/tingyuli/GW-from-goldstone/data/ratio_gamma.npy", data)

    return 0

if __name__ == '__main__':
    energy_ratio_data()