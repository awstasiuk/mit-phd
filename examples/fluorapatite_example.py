import numpy as np

import matplotlib.pyplot as plt

from timeit import default_timer as timer

from nmresearch import Crystal
from nmresearch import Disorder
from nmresearch import Atom, AtomPos
from sklearn.mixture import GaussianMixture


# generate the FaP crystal
d1 = 0.36853
d2 = 0.39785
fl = Atom(dim_s=2, gamma=251.662 * 10**6, name="flourine")
ph = Atom(dim_s=2, gamma=108.291 * 10**6, name="phosphorous")

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
# crystal object
fp_xtal = Crystal(unit_cell, fp_lat)
# disorder computation vehicle, with 5 unitcell shells
mycalc = Disorder(fp_xtal, 5)

# set a central "test-spin"
orig_atom = AtomPos.create_from_atom(atom=fl, position=[0, 0, 0.25 * 6.887])

# set a secondary "test-spin" for correlations
second_pos = np.matmul(fp_lat, [-1, 1, 0.75])
second_atom = AtomPos.create_from_atom(atom=fl, position=second_pos)

# analytically predict the variance for gaussian estimate
v = mycalc.variance_estimate(orig_atom)
def gauss(x):
    return np.exp(-0.5 * x**2 / v) / (2 * v * np.pi) ** 0.5

# some more descriptive statistics analytically
print("Standard deviation of central spin, krad/s: " + str(v**0.5 * 1e-3))
vs = mycalc.variance_estimate(second_atom)
print("Standard deviation of secondary spin, krad/s: " + str(vs**0.5 * 1e-3))
k = mycalc.kurtosis_estimate(orig_atom)
print("Kurtosis of distribution: " + str(k))

# Monte Carlo to generate the distribution
start = timer()
my_distro = mycalc.simulation(orig_atom, 10000, "fp_example.dat")
end = timer()
print("computation time " + str(end - start))

# fit distribution to a 4-mean Gaussian mixture model
xg = my_distro.reshape(-1, 1)
gmm4 = GaussianMixture(n_components=4).fit(xg)
x = np.linspace(-20000, 20000, 1000)
logprob = gmm4.score_samples(x.reshape(-1, 1))
pdf = np.exp(logprob)
plt.hist(
    [y / 1000 for y in my_distro],
    bins=250,
    density=True,
    label="Monte-Carlo \n Simulation",
)
plt.plot(
    x / 1000, gauss(x) * 1000, color="green", label="Gaussian fit \n analytic"
)
plt.plot(
    x / 1000,
    pdf * 1000,
    color="black",
    linestyle="dashed",
    label="4-Gaussian Mix \n of simulation",
)
plt.xlabel("Disordered field, krad/s")
plt.ylabel("Probability")
plt.legend()
plt.show()

# A gaussian with arbitrary variance
def gauss_v(x, va):
    return np.exp(-0.5 * x**2 / va) / (2 * va * np.pi) ** 0.5


print(gmm4.means_)
pos = np.matmul(fp_lat, [d1, d2, 0.25])
disorder_atom = AtomPos.create_from_atom(atom=ph, position=pos)
b111 = np.matmul(fp_lat, [1, 1, 1])
a = 9.375



# Spin diffusion computations
def precision_round(number, digits=3):
    power = "{:e}".format(number).split("e")[1]
    return round(number, -(int(power) - digits))


d001 = mycalc.spin_diffusion_coeff(orig_atom, bdir=[0, 0, 1], a=a)
d001_2 = mycalc.spin_diffusion_second_order(orig_atom, bdir=[0, 0, 1], a=a)
print("Fluorine 0th order term, 001: D = " + str(d001) + " cm^2/s")
print("Fluorine 2nd order term, 001: D = " + str(d001_2) + " cm^2/s")

d111 = mycalc.spin_diffusion_coeff(orig_atom, bdir=b111, a=a)
d111_2 = mycalc.spin_diffusion_second_order(orig_atom, bdir=b111, a=a)
print("Fluorine 0th order term, 111: D = " + str(d111) + " cm^2/s")
print("Fluorine 2nd order term, 111: D = " + str(d111_2) + " cm^2/s")

print("Fluorine Ratio D001/D111 = " + str((d001 + d001_2) / (d111 + d111_2)))

start = timer()
d_tot001 = mycalc.tot_estimated_spin_diffusion_coeff(
    orig_atom, bdir=[0, 0, 1], a=a
)
print(
    "Fluorine Total Estimation 001: "
    + str(precision_round(d_tot001))
    + " cm^2/s"
)
d_tot111 = mycalc.tot_estimated_spin_diffusion_coeff(orig_atom, bdir=b111, a=a)
end = timer()
print(
    "Fluorine Total Estimation 111: "
    + str(precision_round(d_tot111))
    + " cm^2/s"
)
print("Fluorine Ratio D001/D111 = " + str(d_tot001 / d_tot111))
print("Computation time of total estimates: " + str(end - start))


average_remainder = mycalc.average_remainder(
    7, disorder_atom, bdir=[0, 0, 1], a=a
)
print("Remainder for 7 terms 001: " + str(average_remainder))
average_remainder = mycalc.average_remainder(7, disorder_atom, bdir=b111, a=a)
print("Remainder for 7 terms 111: " + str(average_remainder))

d001 = mycalc.spin_diffusion_coeff(disorder_atom, bdir=[0, 0, 1], a=a)
d001_2 = mycalc.spin_diffusion_second_order(disorder_atom, bdir=[0, 0, 1], a=a)
print("0th order term, 001: D = " + str(d001) + " cm^2/s")
print("2nd order term, 001: D = " + str(d001_2) + " cm^2/s")

d111 = mycalc.spin_diffusion_coeff(disorder_atom, bdir=b111, a=a)
d111_2 = mycalc.spin_diffusion_second_order(disorder_atom, bdir=b111, a=a)
print("0th order term, 111: D = " + str(d111) + " cm^2/s")
print("2nd order term, 111: D = " + str(d111_2) + " cm^2/s")

print("Ratio D001/D111 = " + str((d001 + d001_2) / (d111 + d111_2)))

start = timer()
d_tot001 = mycalc.tot_estimated_spin_diffusion_coeff(
    disorder_atom, bdir=[0, 0, 1], a=a
)
print("Total Estimation 001: " + str(precision_round(d_tot001)) + " cm^2/s")
d_tot111 = mycalc.tot_estimated_spin_diffusion_coeff(
    disorder_atom, bdir=b111, a=a
)
end = timer()
print("Total Estimation 111: " + str(precision_round(d_tot111)) + " cm^2/s")
print("Ratio D001/D111 = " + str(d_tot001 / d_tot111))
print("Computation time of total estimates: " + str(end - start))
