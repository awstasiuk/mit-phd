from functools import cache
import rustworkx as rx
import numpy as np
import scipy as sp

from numpy.random import rand


class ClassicalGraph:
    """
    A description of what this class does
    """

    def __init__(self):
        self.base_graph = None

    def generate_percolation_graph(p, generator, dim):
        r"""
        Generates a percolation graph with bond acceptance probability p,
        with the dense graph being defined by the function `generator`. This
        function will be called with the `dim` keyword. That seems like silly
        design? Maybe? The idea is that the structure is defined by `generator`,
        and the size of the structure is defined by `dim`. Obviously, `dim` and
        `generator` need to play nicely together so this maybe isn't very
        end-user friendly.
        """
        G = generator(dim).copy()
        marked_edges = [edge if p < rand() else -1 for edge in G.edge_list()]
        remove_list = (
            [x for x in marked_edges if x != -1] if -1 in marked_edges else marked_edges
        )
        G.remove_edges_from(remove_list)
        return G

    def avg_cluster_size(self, p, dim, generator, repititions=100):
        r"""
        For a square graph of L*L nodes, compute the average cluster size,
        excluding the largest cluster. As L->infinity, this ensures that this
        function diverges at the phase transition, but nowhere else.
        """
        try:
            G0 = generator(dim)
        except Exception as e:
            print("generator and dim are mismatched")
            return
        sites = G0.num_nodes()
        counts = np.zeros(sites + 1, dtype=np.int64)
        for _ in range(repititions):
            G = self.generate_percolation_graph(p, generator, dim)
            comps = rx.connected_components(G)
            distro = [len(comp) for comp in comps]
            largest_local = np.max(distro)
            temp_counts = np.bincount(
                [len(comp) for comp in comps], minlength=sites + 1
            )
            temp_counts[largest_local] = 0
            counts += temp_counts
        bins = np.array(range(sites + 1))

        return (bins**2 @ counts) / (bins @ counts) if bins @ counts != 0 else 0

    def percolation_strength(self, p, dim, generator, repititions=100):
        r"""
        For a square graph of L*L nodes, compute the percolation strength,
        which is the proportion of the graph belonging to the largest cluster,
        which should be the order paramater of the phase tranistion as L->infinity
        """
        P = 0
        try:
            G0 = generator(dim)
        except Exception as e:
            print("generator and dim are mismatched")
            return
        sites = G0.num_nodes()
        for _ in range(repititions):
            G = self.generate_percolation_graph(p, generator, dim)
            distro = [len(comp) for comp in rx.connected_components(G)]
            largest_local = np.max(distro)
            P += largest_local / sites
        return P / repititions

    def cluster_size_and_strength(self, p, dim, generator, repititions=100):
        r"""
        return both the percolation strength and the average cluster size
        for the square L*L percolation graph as a tuple (size, strength)
        """
        try:
            G0 = generator(dim)
        except Exception as e:
            print("generator and dim are mismatched")
            return
        sites = G0.num_nodes()
        counts = np.zeros(sites + 1, dtype=np.int64)
        P = 0
        for _ in range(repititions):
            G = self.generate_percolation_graph(p, generator, dim)
            distro = [len(comp) for comp in rx.connected_components(G)]
            largest_local = np.max(distro)
            P += largest_local / sites
            temp_counts = np.bincount(
                [len(comp) for comp in rx.connected_components(G)], minlength=sites + 1
            )
            temp_counts[largest_local] = 0
            counts += temp_counts
        bins = np.array(range(sites + 1))
        return (bins**2 @ counts) / (bins @ counts), P


class QuantumGraph:
    """
    A description of what this class does
    """

    def __init__(self, base_graph):
        self.base_graph = base_graph

    @cache
    def mean_hopping_prob(sigma, J=1):
        y = J / (4 * sigma)
        return y * np.sqrt(np.pi) * np.exp(y**2) * sp.special.erfc(y)

    def generate_percolation_graph(self, sigma, dim, layers):
        r"""
        Generates a percolation graph with disorder strength sigma, the std
        of a normal distribution.

        The dense graph is defined by the function `generator`. This
        function will be called with the `dim` keyword to determine its size, so these
        should be paired together.
        """
        G = self.base_graph.generate(
            dim, layers
        ).copy()  # avoid potentially mutating the original

        # get each edge's coupling strength, J, and compute the mean hopping prob
        # for the given sigma. Then, randomly keep the bond with that probability, otherwise
        # we mark it for deletion.
        remove_list = [
            edge
            for edge, J in zip(G.edge_list(), G.edges())
            if self.mean_hopping_prob(sigma, J) < rand()
        ]

        # remove and return
        G.remove_edges_from(remove_list)
        return G

    def avg_cluster_size(self, sigma, repititions, dim, layers):
        """


        Args:
            sigma (double): _description_
            repititions (int): _description_
            dim (int): _description_
            layers (int): _description_

        Returns:
            double: _description_
        """
        try:
            G0 = self.base_graph.generate(dim, layers)
        except Exception as e:
            print("generator-param combo is invalid")
            return

        sites = G0.num_nodes()
        counts = np.zeros(sites + 1, dtype=np.int64)
        for _ in range(repititions):
            G = self.generate_percolation_graph(sigma, dim)
            comps = rx.connected_components(G)
            distro = [len(comp) for comp in comps]
            largest_local = np.max(distro)
            temp_counts = np.bincount(
                [len(comp) for comp in comps], minlength=sites + 1
            )
            temp_counts[largest_local] = 0
            counts += temp_counts
        bins = np.array(range(sites + 1))
        return (bins**2 @ counts) / (bins @ counts) if bins @ counts != 0 else 0

    def percolation_strength(self, sigma, repititions, dim, layers):
        r"""
        thots
        """
        P = 0
        try:
            G0 = self.base_graph.generate(dim, layers)
        except Exception as e:
            print("generator-param combo is invalid")
            return
        sites = G0.num_nodes()
        for _ in range(repititions):
            G = self.generate_percolation_graph(sigma, dim)
            distro = [len(comp) for comp in rx.connected_components(G)]
            largest_local = np.max(distro)
            P += largest_local / sites
        return P / repititions
