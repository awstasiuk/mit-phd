import numpy as np
import random


class Atom:
    r"""
    A data structure class for describing an atom.

    """

    def __init__(self, dim_s, gamma, name="atom1", abundance=None):
        r"""
        Initialize the data structure. If there is more than one stable isotope with
        appreciable abundance, `dim_s` and `gamma` should be lists, and `abundance` should
        be specified as a list of probabilities.

        IMPORTANT: Specify `gamma` with units of rad/s/T
        """

        self._dim_s = dim_s
            #int(dim_s) if hasattr(dim_s, "__iter__") else [int(val) for val in dim_s]
        
        self._gamma = gamma
        self._name = name
        self._multi_species = False

        if abundance is not None:
            norm1 = sum(abundance)
            self.abundance=[]
            for x in abundance:
                self._abundance.append(x/norm1)
            self._multi_species = True
            # due to float precision this line could be an issue (maybe normalize the list??)

        if self._multi_species:
            assert len(dim_s) == len(gamma) and len(gamma) == len(abundance)

    @property
    def dim_s(self):
        return self._dim_s
    
    def get_dim(self):
        return self._dim_s

    @property
    def gamma(self):
        return self._gamma
    
    def get_gamma(self):
        return self._gamma

    def multi_species(self):
        return self._multi_species
    
    def name(self):
        return self._name

    @property
    def abundance(self):
        return self._abundance if self.multi_species else [1]

class AtomPos(Atom):
    def __init__(self, pos, atom =None, dim_s=2, gamma=0, name="atom1", cpl=0):
        if atom is None:
            super().__init__(dim_s, gamma, name, abundance = None)
        else:
            if atom.multi_species():
                u = random.uniform(0,1)
                counter=0
                index=-1
                while u >= counter:
                    index+=1
                    counter+=abundance[index]
                mys = atom.get_dim()[index]
                mygamma = atom.get_gamma()[index]
                super().__init__(mys, mygamma, atom.name(), abundance = None)
            else:
                super().__init__(atom.get_dim(), atom.get_gamma(), atom.name(), abundance = None)
        self._pos = pos
        self._cpl=cpl

    
    def pos(self):
        return self._pos
    
    def cpl(self):
        return self._cpl
    
    def set_cpl(self, cpl):
        self._cpl =cpl
        
        
        
    
        
    