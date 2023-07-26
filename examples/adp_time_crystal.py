import numpy as np

import matplotlib.pyplot as plt

from timeit import default_timer as timer
import sys

# sys.path.append("../nmresearch/crystal")
from nmresearch import Crystal
from nmresearch import Disorder
from nmresearch import Atom, AtomPos
from sklearn.mixture import GaussianMixture

x = 0.862604
y = 0.406738
z = 0.123382
xh = 0.894624
yh = 0.487048
zh = 0.837948
w = 0.866002
ph = Atom(dim_s=2, gamma=108.291 * 10**6, name="phosphorous")
h = Atom(dim_s=2, gamma=267.513 * 10**6, name="hydrogen")
# o = Atom(dim_s=6, gamma=-36.274 * 10**6, name="oxygen")
o = Atom(dim_s=1, gamma=0, name="oxygen")
n = Atom(
    dim_s=[2, 3],
    gamma=[19.332 * 10**6, -27.120 * 10**6],
    name=["nitrogen", "nitrogen-15"],
    abundance=[0.9964, 0.0036],
)
# Crystal structure data given by the materials project website of Berkeley Laboratories
unit_cell = {
    ph: np.array(
        [
            [0, 0, 0],
            [0, 1 / 2, 1 / 4],
            [1 / 2, 1 / 2, 1 / 2],
            [1 / 2, 0, 3 / 4],
        ]
    ),
    n: np.array(
        [
            [0, 0, 1 / 2],
            [0, 1 / 2, 3 / 4],
            [1 / 2, 1 / 2, 0],
            [1 / 2, 0, 3 / 4],
        ]
    ),
    o: np.array(
        [
            [x, y, z],
            [1 - x, y + 1 / 2, 1 / 4 - z],
            [y, 1 - x, 1 - z],
            [1 - y, 3 / 2 - x, 1 / 4 + z],
            [x, 1 / 2 - y, 1 / 4 - z],
            [1 - x, 1 - y, z],
            [1 - y, x, 1 - z],
            [y, x - 1 / 2, z + 1 / 4],
            [-1 / 2 + x, y + 1 / 2, z + 1 / 2],
            [3 / 2 - x, y, 3 / 4 - z],
            [y + 1 / 2, 3 / 2 - x, 1 / 2 - z],
            [1 / 2 - y, 1 - x, 3 / 4 + z],
            [x - 1 / 2, 1 - y, 3 / 4 - z],
            [3 / 2 - x, 1 / 2 - y, z + 1 / 2],
            [1 / 2 - y, x - 1 / 2, 1 / 2 - z],
            [y + 1 / 2, x, z + 3 / 4],
        ]
    ),
    h: np.array(
        [
            [xh, yh, zh],
            [1 - xh, yh + 1 / 2, 5 / 4 - zh],
            [yh, 1 - xh, 1 - zh],
            [1 - yh, 3 / 2 - xh, -3 / 4 + zh],
            [xh, 1 / 2 - yh, 5 / 4 - zh],
            [1 - xh, 1 - yh, zh],
            [1 - yh, xh, 1 - zh],
            [yh, xh - 1 / 2, zh - 3 / 4],
            [-1 / 2 + xh, yh + 1 / 2, zh - 1 / 2],
            [3 / 2 - xh, yh, 7 / 4 - zh],
            [yh + 1 / 2, 3 / 2 - xh, 3 / 2 - zh],
            [1 / 2 - yh, 1 - xh, -1 / 4 + zh],
            [xh - 1 / 2, 1 - yh, 7 / 4 - zh],
            [3 / 2 - xh, 1 / 2 - yh, zh - 1 / 2],
            [1 / 2 - yh, xh - 1 / 2, 3 / 2 - zh],
            [yh + 1 / 2, xh, zh - 1 / 4],
            [w, 1 / 4, 1 / 8],
            [w - 1 / 2, 3 / 4, 5 / 8],
            [1 / 4, 1 - w, 7 / 8],
            [3 / 4, 3 / 2 - w, 3 / 8],
            [1 - w, 3 / 4, 1 / 8],
            [3 / 2 - w, 1 / 4, 5 / 8],
            [1 / 4, w - 1 / 2, 3 / 8],
            [3 / 4, w, 7 / 8],
        ]
    ),
}

adp_lat = np.array(
    [
        [7.74, 0, 0],
        [0, 7.74, 0],
        [0, 0, 7.04],
    ]
)
adp_xtal = Crystal(unit_cell, adp_lat)
mycalc = Disorder(adp_xtal)
orig_atom = AtomPos.create_from_atom(
    atom=ph, position=mycalc.crystal.to_real_space([0, 0, 0.25 * 7.04])
)

v = mycalc.variance_estimate(orig_atom)


def gauss(x):
    return np.exp(-0.5 * x**2 / v) / (2 * v * np.pi) ** 0.5


print(v**0.5)
k = mycalc.kurtosis_estimate(orig_atom)
print(k)
print(mycalc.mean_field_calc(orig_atom))
start = timer()
my_distro = mycalc.simulation(orig_atom, 5000, "adp_example_new.dat")
end = timer()
print("computation time " + str(end - start))
xg = my_distro.reshape(-1, 1)
gmm3 = GaussianMixture(n_components=3).fit(xg)
x = np.linspace(-200000, 200000, 10000)
logprob = gmm3.score_samples(x.reshape(-1, 1))
pdf = np.exp(logprob)
plt.hist(my_distro, bins=250, density=True, label="Monte-Carlo \n Simulation")
plt.plot(x, gauss(x), color="green", label="Gaussian fit \n analytic")
plt.plot(
    x,
    pdf,
    color="black",
    linestyle="dashed",
    label="3-Gaussian Mix \n of simulation",
)
plt.show()
print(gmm3.means_)
