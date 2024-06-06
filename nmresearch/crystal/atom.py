from random import uniform


class Atom:
    r"""
    A data structure class for describing an atom. It contains relevant species level
    information for an atomic nuclear spin. This class contains the spin dimension of
    the nuclear spin, `dim_s`, a descriptive name of the spin, `name`, and the
    gyromagnetic ration of the spin, `gamma` given in units of rad/s/T.

    If there is more than one stable isotope with ppreciable abundance, `dim_s`, `name`,
    and `gamma` should be given lists, and `abundance` must be specified as a list of probabilities
    for the presence of each stable isotope.

    IMPORTANT: Failure to specify `gamma` with units of rad/s/T will result in incorrect numerical
    values
    """

    def __init__(self, dim_s, gamma, name, abundance=None):

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

    def __key(self):
        return (self.dim_s, self.gamma, self.name)

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.__key() == other.__key()
        return False

    def __hash__(self):
        return hash(self.__key())


class AtomPos(Atom):
    r"""
    Inherits properties from Atom class, plus has a set position and a user defined coupling strength.
    No abundance probabilities, must be a fixed atom in space.
    Can be generated using AtomPos call with all arguments, or with create_from_atom
    which takes a pre-defined atom and a position
    """

    def __init__(self, dim_s, position, gamma=0, name="atom1", coupling=0):
        super().__init__(dim_s, gamma, name, abundance=None)
        self._position = position
        self._coupling = coupling

    @property
    def position(self):
        return self._position

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, value):
        self._coupling = value

    @staticmethod
    def create_from_atom(atom, position, coupling=0):
        """
        Intended instantiation method for a positional atom. This function accepts an `Atom`
        and a position, and uses this information to create a positional atom. If atom has multiple
        species, we choose one randomly according to abundance probabilities defined in the `atom`
        argument.
        """
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

        return AtomPos(mys, position, mygamma, myname, coupling)
