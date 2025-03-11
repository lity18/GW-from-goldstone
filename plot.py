import numpy as np
from mpi4py import MPI
import time
import os
import datetime
import matplotlib.pyplot as plt
import scipy.special as spec
import scipy as sci 
import function as f


def ratio_gamma():
    gamma0 = np.array([5, 10, 15, 20])
    phase0 = np.array([1, 6, 9, 12])
    data = np.load("/home/storage0/users/tingyuli/GW-from-goldstone/data/ratio_gamma.npy")
    plt.rc('text', usetex=True)
    width = 1
    for i in range(np.size(phase0)):
        plt.plot(gamma0, data[i, 1:, 0], label=r"$\theta=\frac{" + str(phase0[i]) + r"}{16}\pi$", linewidth=width)
    plt.legend()
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$r_{r}=\frac{E_r}{E_a}$")
    plt.savefig("/home/storage0/users/tingyuli/GW-from-goldstone/fig/ratio_gamma.png", dpi=1000)
    return 0


if __name__ == '__main__':
    ratio_gamma()