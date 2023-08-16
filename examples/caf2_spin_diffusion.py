#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:26:26 2023

@author: garrettheller
"""

import numpy as np

import matplotlib.pyplot as plt

from timeit import default_timer as timer

from nmresearch import Crystal
from nmresearch import Disorder
from nmresearch import Atom, AtomPos


def precision_round(number, digits=3):
    power = "{:e}".format(number).split("e")[1]
    return round(number, -(int(power) - digits))


a = 5.463
fl = Atom(dim_s=2, gamma=251.662 * 10**6, name="flourine")
ca = Atom(
    dim_s=[1, 8],
    gamma=[0, 18.727 * 10**6],
    name=["calcium-40", "calcium-43"],
    abundance=[0.99865, 0.00135],
)

unit_cell = {
    ca: np.array(
        [[0, 0, 0], [0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [0, 1 / 2, 1 / 2]]
    ),
    fl: np.array(
        [
            [1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 4, 3 / 4],
            [1 / 4, 3 / 4, 1 / 4],
            [3 / 4, 1 / 4, 1 / 4],
            [1 / 4, 3 / 4, 3 / 4],
            [3 / 4, 1 / 4, 3 / 4],
            [3 / 4, 3 / 4, 1 / 4],
            [3 / 4, 3 / 4, 3 / 4],
        ]
    ),
}

caf2_lat = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
caf2_xtal = Crystal(unit_cell, caf2_lat)
mycalc = Disorder(caf2_xtal, 5)
orig_atom = AtomPos.create_from_atom(
    atom=fl, position=[-0.25 * a, 0.25 * a, 0.25 * a]
)
second_atom = AtomPos.create_from_atom(
    atom=fl, position=[1.25 * a, -1.25 * a, 0.75 * a]
)


def gauss_v(x, va):
    return np.exp(-0.5 * x**2 / va) / (2 * va * np.pi) ** 0.5


r"""
my_atoms = caf2_xtal.generate_lattice(1)
for atompos in my_atoms:
    if atompos.name == "flourine":
        v0 = mycalc.homonuclear_double_variance_estimate(orig_atom, atompos)
        print("Minus variance krad^2/s^2: " + str(v0 * 1e-6))
"""

double_distro_minus = mycalc.homonuclear_double_simulation(
    orig_atom,
    second_atom,
    trials=2000,
    bdir=[0, 0, 1],
    filename="bijdistro.dat",
    regen=True,
)
v1 = mycalc.homonuclear_double_variance_estimate(
    orig_atom, second_atom, bdir=[0, 0, 1]
)
x = np.linspace(-800000, 800000, 1000)
plt.hist(
    double_distro_minus, bins=250, density=True, label="Double \n Simulation"
)
plt.plot(x, gauss_v(x, v1), color="green", label="Gaussian fit \n analytic")
plt.xlabel("rad/s")
plt.show()

hbar = 1.05457 * 10 ** (-34)
# average_remainder = mycalc.average_remainder(7, orig_atom, bdir=[0, 0, 1], a=a)
# print("Remainder for 7 terms 001: " + str(average_remainder))
# average_remainder = mycalc.average_remainder(7, orig_atom, bdir=[1, 1, 1], a=a)
# print("Remainder for 7 terms 111: " + str(average_remainder))

d001 = mycalc.spin_diffusion_coeff(orig_atom, bdir=[0, 0, 1], a=a)
d001_2 = mycalc.spin_diffusion_second_order(orig_atom, bdir=[0, 0, 1], a=a)
print("0th order term, 001: D = " + str(d001) + " cm^2/s")
print("2nd order term, 001: D = " + str(d001_2) + " cm^2/s")

d111 = mycalc.spin_diffusion_coeff(orig_atom, bdir=[1, 1, 1], a=a)
d111_2 = mycalc.spin_diffusion_second_order(orig_atom, bdir=[1, 1, 1], a=a)
print("0th order term, 111: D = " + str(d111) + " cm^2/s")
print("2nd order term, 111: D = " + str(d111_2) + " cm^2/s")

print("Ratio D001/D111 = " + str((d001 + d001_2) / (d111 + d111_2)))

start = timer()
d_tot001 = mycalc.tot_estimated_spin_diffusion_coeff(
    orig_atom, bdir=[0, 0, 1], a=a
)
print("Total Estimation 001: " + str(precision_round(d_tot001)) + " cm^2/s")
d_tot111 = mycalc.tot_estimated_spin_diffusion_coeff(
    orig_atom, bdir=[1, 1, 1], a=a
)
end = timer()
print("Total Estimation 111: " + str(precision_round(d_tot111)) + " cm^2/s")
print("Ratio D001/D111 = " + str(d_tot001 / d_tot111))
print("Computation time of total estimates: " + str(end - start))
