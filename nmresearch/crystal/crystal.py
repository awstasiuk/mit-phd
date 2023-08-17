from nmresearch.crystal.atom import AtomPos

from numpy import array, matmul


class Crystal:
    r"""
    This class describes a crystal structure. A crystal has two important properties,
    the `unit_cell` (conventional unit cell) and the `lattice_vecs` (lattice vectors). 
    We want to take a convential unit cell, which will generally contain more than one atom. 
    Each `Atom` in the unit cell contains positional data, and instrinsic coupling strength data.

    `lattice_vecs` should be a 3 dimensional real matrix consisting of the primitive lattice vectors
    for the crystal, as columns of the matrix.
    """

    def __init__(self, unit_cell, lattice_vecs):
        self._unit_cell = unit_cell
        self._lattice_vecs = lattice_vecs

    def generate_lattice(self, shell_radius=1):
        r"""
        Generates a list of points for each `Atom` species in the lattice. `shell_radius`
        is the number of layers of conventional unit cells surrounding the central unit cell,
        so that the default `shell_radius` of 1 returns a lattice formed by a cube of
        9 unit cells (the core and one layer).
        """
        shell_arr = []
        for a in range(-shell_radius, shell_radius + 1):
            for b in range(-shell_radius, shell_radius + 1):
                for c in range(-shell_radius, shell_radius + 1):
                    shell_arr.append(array([a, b, c]))

        lattice = []
        for atoms in self.unit_cell:
            for pos in self.unit_cell[atoms]:
                for x in shell_arr:
                    newpos = self.to_real_space(pos + x)
                    lattice.append(
                        AtomPos.create_from_atom(atom=atoms, position=newpos)
                    )

        return lattice

    def to_real_space(self, vec):
        """
        Matrix converter between reduced coordinate system and the real space coordinates
        """
        return matmul(self._lattice_vecs, vec)

    @property
    def unit_cell(self):
        return self._unit_cell

    @property
    def lattice_vecs(self):
        return self._lattice_vecs
