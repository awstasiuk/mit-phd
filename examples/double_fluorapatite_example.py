import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from nmresearch import Crystal
from nmresearch import Disorder
from nmresearch import Atom, AtomPos

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
fp_xtal = Crystal(unit_cell, fp_lat)
mycalc = Disorder(fp_xtal, 3)
orig_atom = AtomPos.create_from_atom(atom=fl, position=[0, 0, 0.25 * 6.887])
second_pos = np.matmul(fp_lat, [-1, 1, 0.75])
second_atom = AtomPos.create_from_atom(atom=fl, position=second_pos)

v = mycalc.variance_estimate(orig_atom)


def gauss(x):
    return np.exp(-0.5 * x**2 / v) / (2 * v * np.pi) ** 0.5


vs = mycalc.variance_estimate(second_atom)
print("Original atom variance, Mrad/s: " + str(v * 1e-6))
print("Secondary atom variance, Mrad/s: " + str(vs * 1e-6))
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


def gauss_v(x, va):
    return np.exp(-0.5 * x**2 / va) / (2 * va * np.pi) ** 0.5


x = np.linspace(-20000, 20000, 1000)
print("=====\n Now the two-atom experiment \n===== ")
v2 = mycalc.double_variance_estimate(
    orig_atom, second_atom, False
)  # Calculate variance of difference of disordered fields at two atoms
print("Minus variance Mrad/s: " + str(v2 * 1e-6))
v1 = mycalc.double_variance_estimate(
    orig_atom, second_atom, True
)  # Calculate variance of sum of difference of disordered fields at two atoms
print("Plus variance Mrad/s: " + str(v1 * 1e-6))

double_distro_plus, double_distro_minus = mycalc.double_simulation(
    orig_atom, second_atom, 2000, ("distro_plus4.dat", "distro_minus4.dat")
)
plt.hist(
    double_distro_plus, bins=250, density=True, label="Double \n Simulation"
)
plt.plot(x, gauss_v(x, v1), color="green", label="Gaussian fit \n analytic")
plt.show()
plt.hist(
    double_distro_minus, bins=250, density=True, label="Double \n Simulation"
)
plt.plot(x, gauss_v(x, v2), color="green", label="Gaussian fit \n analytic")
plt.show()

print("\n ========================\n")

my_atoms = fp_xtal.generate_lattice(1)
for atompos in my_atoms:
    if atompos.name == "flourine":
        v0 = mycalc.double_variance_estimate(orig_atom, atompos, False)
        print("Minus variance Mrad/s: " + str(v0 * 1e-6))

for atompos in my_atoms:
    if atompos.name == "flourine":
        v0 = mycalc.double_variance_estimate(orig_atom, atompos, True)
        print("Plus variance Mrad/s: " + str(v0 * 1e-6))
