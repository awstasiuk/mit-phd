import numpy as np


class Atom:
    r"""
    A data structure class for describing an atom.

    """

    def __init__(self, dim_s, gamma, abundance=None):
        r"""
        Initialize the data structure. If there is more than one stable isotope with
        appreciable abundance, `dim_s` and `gamma` should be lists, and `abundance` should
        be specified as a list of probabilities.

        IMPORTANT: Specify `gamma` with units of rad/s/T
        """

        self._dim_s = (
            int(dim_s) if hasattr(dim_s, "__iter__") else [int(val) for val in dim_s]
        )
        self._gamma = gamma
        self._multi_species = False

        if abundance is not None:
            self._abundance = abundance
            self._multi_species = True
            # due to float precision this line could be an issue
            assert sum(abundance) == 1

        if self._multi_species:
            assert len(dim_s) == len(gamma) and len(gamma) == len(abundance)

    @property
    def dim_s(self):
        return self._dim_s

    @property
    def gamma(self):
        return self._gamma

    @property
    def multi_species(self):
        return self._multi_species

    @property
    def abundance(self):
        return self._abundance if self.multi_species else [1]
