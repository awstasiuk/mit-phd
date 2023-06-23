from random import uniform


class Atom:
    r"""
    A data structure class for describing an atom.

    """

    def __init__(self, dim_s, gamma, name, abundance=None):
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
    def __init__(self, dim_s, position, gamma=0, name="atom1", couple=0):
        super().__init__(dim_s, gamma, name, abundance=None)
        self._position = position
        self._couple = couple

    @property
    def position(self):
        return self._position

    def couple(self):
        return self._couple

    def set_couple(self, couple):
        self._couple = couple

    @staticmethod
    def create_from_atom(atom, position, couple=0):
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
        else:
            mys = atom.dim_s
            mygamma = atom.gamma
            myname = atom.name

        return AtomPos(mys, position, mygamma, myname, couple)
