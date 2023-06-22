from random import uniform


class Atom:
    r"""
    A data structure class for describing an atom.

    """

    def __init__(self, dim_s, gamma, name, abundance=None, pos=None, cpl=0):
        r"""
        Initialize the data structure. If there is more than one stable isotope with
        appreciable abundance, `dim_s` and `gamma` should be lists, and `abundance` should
        be specified as a list of probabilities.

        IMPORTANT: Specify `gamma` with units of rad/s/T
        """

        self._dim_s = dim_s

        self._gamma = gamma
        self._name = name
        self._multi_species = False

        if abundance is not None:
            norm1 = sum(abundance)
            self._abundance = [x / norm1 for x in abundance]
            self._multi_species = True

        if self._multi_species:
            assert len(dim_s) == len(gamma) == len(abundance) == len(name)

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
    def name(self):
        return self._name

    @property
    def abundance(self):
        return self._abundance if self.multi_species else [1]


class AtomPos(Atom):
    def __init__(self, dim_s, pos, gamma=0, name="atom1", cpl=0):
        if atom is None:
            super().__init__(dim_s, gamma, name, abundance=None)
        else:
            if atom.multi_species:
                u = uniform(0, 1)
                counter = 0
                index = -1
                while u >= counter:
                    index += 1
                    counter += atom.abundance[index]
                mys = atom.dim_s[index]
                mygamma = atom.gamma[index]
                myname = atom.name[index]
                super().__init__(mys, mygamma, myname, abundance=None)
            else:
                super().__init__(
                    atom.dim_s,
                    atom.gamma,
                    atom.name,
                    abundance=None,
                )
        self._pos = pos
        self._cpl = cpl

    def pos(self):
        return self._pos

    def cpl(self):
        return self._cpl

    def set_cpl(self, cpl):
        self._cpl = cpl

    @staticmethod
    def create_from_atom(atom, position):
        return AtomPos(atom.gamma)