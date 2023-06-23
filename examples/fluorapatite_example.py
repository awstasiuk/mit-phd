from sympy import inverse_fourier_transform, cos, exp, sqrt, pi
from sympy.abc import x, k
import numpy as np

import matplotlib.pyplot as plt

from timeit import default_timer as timer
import sys

sys.path.append("../nmresearch/crystal")
import crystal
import atom
import disorder

import pickle

d1 = 0.36853
d2 = 0.39785
fl = atom.Atom(dim_s=2, gamma=251.662 * 10**6, name="flourine")
ph = atom.Atom(dim_s=2, gamma=108.291 * 10**6, name="phosphorous")

unit_cell = {
    fl: np.array([[0, 0, 1 / 4], [0, 0, 3 / 4]]),
    ph: np.array(
        [
            [d1, d2, 0.25],
            [-d2, d1 - d2, 0.25],
            [d2 - d1, -d1, 0.25],
            [-d1, -d2, 0.75],
            [d2, d2 - d1, 0.75],
            [d1 - d2, d1, 0.75],
        ]
    ),
}

fp_lat = np.array(
    [
        [9.375, 9.375 * np.cos(120 * np.pi / 180), 0],
        [0, 9.375 * np.sin(120 * np.pi / 180), 0],
        [0, 0, 6.887],
    ]
)
fp_xtal = crystal.Crystal(unit_cell, fp_lat)
mycalc = disorder.Disorder(fp_xtal, 3)
orig_atom = atom.AtomPos.create_from_atom(
    atom=fl, position=[0, 0, 0.25 * 6.887]
)

v = mycalc.variance_estimate(orig_atom)


def gauss(x):
    return np.exp(-0.5 * x**2 / v) / (2 * v * np.pi) ** 0.5


print(v)
k = mycalc.kurtosis_estimate(orig_atom)
print(k)
print(mycalc.mean_field_calc(orig_atom))
start = timer()
my_distro = mycalc.simulation(orig_atom, 5000, "fp_example.dat")
end = timer()
print("computation time " + str(end - start))
# xg = my_distro.reshape(-1,1)
# gmm4 = GaussianMixture(n_components=4).fit(xg)
x = np.linspace(-20000, 20000, 1000)
# logprob = gmm4.score_samples(x.reshape(-1, 1))
# pdf = np.exp(logprob)

plt.hist(my_distro, bins=250, density=True, label="Monte-Carlo \n Simulation")
plt.plot(x, gauss(x), color="green", label="Gaussian fit \n analytic")
# plt.plot(x, pdf, color='black', linestyle='dashed', label = '4-Gaussian Mix \n of simulation')
plt.show()
