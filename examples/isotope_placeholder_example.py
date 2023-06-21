from sympy import inverse_fourier_transform, cos, exp, sqrt, pi
from sympy.abc import x, k
import numpy as np

import matplotlib.pyplot as plt
from timeit import default_timer as timer

import crystal
import atom
import disorder

import pickle

d1 = 0.36853
d2 = 0.39785
fl = atom.Atom(dim_s=2, gamma=251.662 * 10**6, name="flourine")

ph2 = atom.Atom(
    dim_s=[2, 1],
    gamma=[108.291 * 10**6, 10**6],
    name=["phosphorous-31", "phosphorous-xx"],
    abundance=[0.8, 0.2],
)

unit_cell_2 = {
    fl: np.array([[0, 0, 1 / 4], [0, 0, 3 / 4]]),
    ph2: np.array(
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

fp_xtal = crystal.Crystal(unit_cell_2, fp_lat)

mycalc = disorder.Disorder(fp_xtal)
origin_atom = atom.AtomPos(atom=fl, pos=[0, 0, 0.25 * 6.887])

v = mycalc.variance_estimate(orig_atom, ph2.name, 5)


def gauss(x):
    return np.exp(-0.5 * x**2 / v) / (2 * v * np.pi) ** 0.5


print(v)
k = mycalc.kurtosis_estimate(orig_atom, atomlis, 5)
print(k)
print(mycalc.mean_field_calc(orig_atom, atomlis, 5))
start = timer()
my_distro = mycalc.simulation(orig_atom, atomlis, 50000, "new_distro.dat", 2)
end = timer()
print("computation time " + str(end - start))
x = np.linspace(-20000, 20000, 1000)

plt.hist(my_distro, bins=250, density=True, label="Monte-Carlo \n Simulation")
plt.plot(x, gauss(x), color="green", label="Gaussian fit \n analytic")
plt.show()
