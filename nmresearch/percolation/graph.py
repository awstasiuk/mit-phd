from functools import cache
import rustworkx as rx
import numpy as np
import scipy as sp

from numpy.random import rand


class ClassicalGraph:
    """
    A description of what this class does
    """

    def __init__(self, base_graph):
        """
        initialize things, doi

        Args:
            base_graph (BaseGraph): A graph wrapper with an implemented `generate`
            function, like FccGraph or GridGraph
        """
        self.base_graph = base_graph

    def generate_percolation_graph(self, p, dim, layers):
        G = self.base_graph.load_graph(
            dim, layers
        ).copy()  # avoid potentially mutating the original

        # if dice roll is less than given p-val, mark bond for removal
        remove_list = [edge for edge, J in zip(G.edge_list(), G.edges()) if p < rand()]

        # remove and return
        G.remove_edges_from(remove_list)
        return G

    def avg_cluster_size(self, p, dim, layers, repititions=100):
        r"""
        For a square graph of L*L nodes, compute the average cluster size,
        excluding the largest cluster. As L->infinity, this ensures that this
        function diverges at the phase transition, but nowhere else.
        """
        try:
            G0 = self.base_graph.load_graph(dim, layers)
        except Exception as e:
            print("generator and dim are mismatched")
            return
        sites = G0.num_nodes()
        counts = np.zeros(sites + 1, dtype=np.int64)
        for _ in range(repititions):
            G = self.generate_percolation_graph(p, dim, layers)
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

    def percolation_strength(self, p, dim, layers, repititions=100):
        r"""
        For a square graph of L*L nodes, compute the percolation strength,
        which is the proportion of the graph belonging to the largest cluster,
        which should be the order paramater of the phase tranistion as L->infinity
        """
        P = 0
        try:
            G0 = self.base_graph.load_graph(dim, layers)
        except Exception as e:
            print("generator and dim are mismatched")
            return
        sites = G0.num_nodes()
        for _ in range(repititions):
            G = self.generate_percolation_graph(p, dim, layers)
            distro = [len(comp) for comp in rx.connected_components(G)]
            largest_local = np.max(distro)
            P += largest_local / sites
        return P / repititions

    def cluster_size_and_strength(self, p, dim, layers, repititions=100):
        r"""
        return both the percolation strength and the average cluster size
        for the square L*L percolation graph as a tuple (size, strength)
        """
        try:
            G0 = self.base_graph.load_graph(dim, layers)
        except Exception as e:
            print("generator and dim are mismatched")
            return
        sites = G0.num_nodes()
        counts = np.zeros(sites + 1, dtype=np.int64)
        P = 0
        for _ in range(repititions):
            G = self.generate_percolation_graph(p, dim, layers)
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
        """
        initialize things, doi

        Args:
            base_graph (BaseGraph): A graph wrapper with an implemented `generate`
            function, like FccGraph or GridGraph
        """
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
        G = self.base_graph.load_graph(
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

    def avg_cluster_size(self, sigma, dim, layers, repititions):
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
            G0 = self.base_graph.load_graph(dim, layers)
        except Exception as e:
            print("generator-param combo is invalid")
            return

        sites = G0.num_nodes()
        counts = np.zeros(sites + 1, dtype=np.int64)
        for _ in range(repititions):
            G = self.generate_percolation_graph(sigma, dim, layers)
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

    def percolation_strength(self, sigma, dim, layers, repititions):
        r"""
        thots
        """
        P = 0
        try:
            G0 = self.base_graph.load_graph(dim, layers)
        except Exception as e:
            print("generator-param combo is invalid")
            return
        sites = G0.num_nodes()
        for _ in range(repititions):
            G = self.generate_percolation_graph(sigma, dim, layers)
            distro = [len(comp) for comp in rx.connected_components(G)]
            largest_local = np.max(distro)
            P += largest_local / sites
        return P / repititions
