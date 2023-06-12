import numpy as np
import scipy as sp


class Crystal:
    r"""
    This class describes a crystal structure. A crystal has two important properties,
    the `unit_cell` and the `lattice` vectors. We want to take a convential unit cell,
    which will contain more than one atom. Each atom in the unit cell should have the data
    which describes it, and we use the `Atom` class as its data structure.

    """

    def __init__(self, unit_cell, lattice):
        self._unit_cell = unit_cell
        self._lattice = lattice

    def generate_crystal(self, shell_radius=1):
        r"""
        Generates a list of points for each `Atom` species in the lattice. `shell_radius`
        describes the number of unit cells
        """
        return None

    @property
    def unit_cell(self):
        return self._unit_cell

    @property
    def lattice(self):
        return self._lattice
