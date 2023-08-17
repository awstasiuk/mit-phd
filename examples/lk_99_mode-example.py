import numpy as np

import matplotlib.pyplot as plt

from timeit import default_timer as timer

from nmresearch import Crystal
from nmresearch import Disorder
from nmresearch import Atom, AtomPos
from sklearn.mixture import GaussianMixture

r"""
Structure given as Cu on Pb(1) for OH columns
On page 11 (appendix) from pre-print by Griffin, 2023
"""

cu = Atom(
    dim_s=[3, 3],
    gamma=[70.965e6, 76.018e6],
    abundance=[0.697, 0.303],
    name=["copper", "copper"],
)

pb = Atom(dim_s=2, gamma=56.4e6, name="lead")
ph = Atom(dim_s=2, gamma=108.291e6, name="phosphorous")
# Oxygen have no nuclear spin so not added
h = Atom(dim_s=2, gamma=267.5105e6, name="hydrogen")
unit_cell = {
    cu: np.array([[0.33333, 0.66667, 0.99594]]),
    pb: np.array(
        [
            [0.66667, 0.33333, 0.97556],
            [0.66667, 0.33333, 0.47828],
            [0.33333, 0.66667, 0.47831],
            [0.24304, 0.99176, 0.23619],
            [0.99642, 0.73675, 0.74019],
            [0.74872, 0.75695, 0.23619],
            [0.26324, 0.25966, 0.74019],
        ]
    ),
    ph: np.array(
        [
            [0.39800, 0.37662, 0.22932],
            [0.59606, 0.63623, 0.76062],
            [0.62337, 0.02138, 0.22932],
            [0.36376, 0.95982, 0.76062],
            [0.97861, 0.60199, 0.22932],
            [0.04017, 0.40393, 0.76062],
        ]
    ),
    h: np.array([[0, 0, 0.445], [0, 0, 0.84006]]),
}

lk99_lat = np.array(
    [
        [9.73753, 0, 0],
        [-4.86786, 8.43294, 0],
        [0, 0, 7.30653],
    ]
)
lk99_xtal = Crystal(unit_cell, lk99_lat)
mycalc = Disorder(lk99_xtal, 4)

pos1 = np.matmul(lk99_lat, [0.39800, 0.37662, 0.22932])
pos2 = np.matmul(
    lk99_lat,
    [0.66667, 0.33333, 0.47828],
)
orig_atom = AtomPos.create_from_atom(atom=ph, position=pos1)

second_atom = AtomPos.create_from_atom(atom=pb, position=pos2)

v = mycalc.variance_estimate(second_atom)

def gauss(x):
    return np.exp(-0.5 * x**2 / v) / (2 * v * np.pi) ** 0.5


# print descriptive statistics about the distribution
vs = mycalc.variance_estimate(orig_atom)
print("Original atom standard deviation, krad/s: " + str(v**0.5 * 1e-3))
print("Secondary atom standard deviation, krad/s: " + str(vs**0.5 * 1e-3))
k = mycalc.kurtosis_estimate(second_atom)
print("Kurtosis of distribution: " + str(k))


# Monte Carlo calculate the distribution itself
start = timer()
my_distro = mycalc.simulation(orig_atom, 5000, "lk99_example.dat")
end = timer()
print("computation time " + str(end - start))


# Plot the generated histogram with a Gaussian fit
xg = my_distro.reshape(-1, 1)
x = np.linspace(-5000, 5000, 1000)
plt.hist(
    my_distro / 1000, bins=250, density=True, label="Monte-Carlo \n Simulation"
)
plt.plot(
    x / 1000, gauss(x) * 1000, color="green", label="Gaussian fit \n analytic"
)
plt.xlabel("krad/s")
plt.legend()
plt.show()


# compute spin diffusion coefficients
def precision_round(number, digits=3):
    power = "{:e}".format(number).split("e")[1]
    return round(number, -(int(power) - digits))


a = 9.73753

b111 = np.matmul(lk99_lat, [1, 1, 1])

d001 = mycalc.spin_diffusion_coeff(orig_atom, bdir=[0, 0, 1], a=a)
d001_2 = mycalc.spin_diffusion_second_order(orig_atom, bdir=[0, 0, 1], a=a)
print("0th order term, 001: D = " + str(d001) + " cm^2/s")
print("2nd order term, 001: D = " + str(d001_2) + " cm^2/s")

d111 = mycalc.spin_diffusion_coeff(orig_atom, bdir=b111, a=a)
d111_2 = mycalc.spin_diffusion_second_order(orig_atom, bdir=b111, a=a)
print("0th order term, 111: D = " + str(d111) + " cm^2/s")
print("2nd order term, 111: D = " + str(d111_2) + " cm^2/s")

print("Ratio D001/D111 = " + str((d001 + d001_2) / (d111 + d111_2)))

start = timer()
d_tot001 = mycalc.tot_estimated_spin_diffusion_coeff(
    orig_atom, bdir=[0, 0, 1], a=a
)
print("Total Estimation 001: " + str(precision_round(d_tot001)) + " cm^2/s")
d_tot111 = mycalc.tot_estimated_spin_diffusion_coeff(orig_atom, bdir=b111, a=a)
end = timer()
print("Total Estimation 111: " + str(precision_round(d_tot111)) + " cm^2/s")
print("Ratio D001/D111 = " + str(d_tot001 / d_tot111))
print("Computation time of total estimates: " + str(end - start))
