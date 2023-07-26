#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:26:26 2023

@author: garrettheller
"""

import numpy as np

import matplotlib.pyplot as plt

from timeit import default_timer as timer
import sys

sys.path.append("../nmresearch/crystal")
from nmresearch import Crystal
from nmresearch import Disorder
from nmresearch import Atom, AtomPos

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
            [1 / 4, 1 / 4, 1 / 4],
        ]
    ),
}

caf2_lat = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
caf2_xtal = Crystal(unit_cell, caf2_lat)
mycalc = Disorder(caf2_xtal, 5)
orig_atom = AtomPos.create_from_atom(
    atom=fl, position=[0.25 * a, 0.25 * a, 0.25 * a]
)
print(
    str(mycalc.spin_diffusion_coeff_parallel(orig_atom, bdir=[1, 1, 1]))
    + " cm^2/s"
)
