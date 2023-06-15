import numpy as np
import scipy as sp
import atom


class Crystal:
    r"""
    This class describes a crystal structure. A crystal has two important properties,
    the `unit_cell` and the `lattice` vectors. We want to take a convential unit cell,
    which will contain more than one atom. Each atom in the unit cell should have the data
    which describes it, and we use the `Atom` class as its data structure.

    """

    def __init__(self, unit_cell, lattice):
        self._unit_cell = unit_cell #dictionary of atom to atom positions in unit cell in cell basis
        self._lattice = lattice #matrix converter between cell basis and real space, typically diagonal

    def generate_crystal(self, shell_radius=1):
        shell_arr=[]
        mylis=[]
        for a in range(-shell_radius, shell_radius+1):
            for b in range(-shell_radius, shell_radius+1):
                for c in range(-shell_radius, shell_radius+1):
                    shell_arr.append(np.array([a,b,c]))
        for atoms in self._unit_cell:
            for pos in self._unit_cell[atoms]:
                for x in shell_arr:
                    newpos = self.to_real_space(pos + x)
                    mylis.append(atom.AtomPos(atom = atoms, pos = newpos))
        return mylis
                
# =============================================================================
#         d=dict()
#         for atoms in self._unit_cell:
#             d[atoms] = np.concatenate([self.to_real_space(x + y) for x in shell_arr for y in self._unit_cell[atoms]])
#         return d
# =============================================================================
            
        #return np.concatenate(self.shift(x) for x in shell_arr)
        r"""
        Generates a list of points for each `Atom` species in the lattice. `shell_radius`
        describes the number of unit cells
        """
        return None

    @property
    def unit_cell(self):
        return self._unit_cell
    #unit cell a dictionary of atom: nd array of positions

    @property
    def lattice(self):
        return self._lattice
    
# =============================================================================
#     def shift(self, shift_vec):
#         return np.array([(atompos[0], self.to_real_space(atompos[1] + shift_vec)) for atompos in self._unit_cell])
# =============================================================================
    
    def to_real_space(self, vec):
        return np.matmul(self._lattice, vec)
    
# =============================================================================
#     def to_real_space(self, AtomPos):
#         return AtomPos(np.matmul(self._lattice, AtomPos.pos()))
# =============================================================================
