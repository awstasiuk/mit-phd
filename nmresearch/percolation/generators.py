from functools import cache
import pickle
import numpy as np
import rustworkx as rx


class BaseGraph:

    def __init__(self):
        # do something ?
        self.G = None

    @cache
    def load_graph(fmt, generator, dim):
        try:
            G = pickle.load(open(f"{fmt}{dim}.dat", "rb"))
        except (OSError, IOError) as e:
            G = generator(dim)
            with open(f"{fmt}{dim}.dat", "wb") as fi:
                pickle.dump(G, fi)
        return G

    @staticmethod
    def generate_fcc(dim, weight_adj):
        """
        Generate a 3D fcc graph of size `dim` of Manhattan radius dim
        edges and their weights are defined by the `weight_adj` structure.


        Args:
            dim (int): _description_
            weight_adj (list): a list of list pairs of format [J, adjacency]
        """
        G = rx.PyGraph(multigraph=False)
        f_atom_pos = BaseGraph._lif_f_sublattice(dim)
        node_indices = G.add_nodes_from(f_atom_pos)

        for J, adj_layer in weight_adj:
            for idxA in node_indices:
                source = G.get_node_data(idxA)
                counter = 0
                for idxB in node_indices:
                    target = G.get_node_data(idxB)
                    if np.any(np.all(adj_layer == source - target, axis=1)):
                        G.add_edge(idxA, idxB, J)
                        counter += 1
                    if counter == len(adj_layer):
                        break

        return G

    def _lif_f_sublattice(r0):
        r"""
        Generate an array containing all of the positions of fluorine atoms in a LiF lattice which
        extends from -r0 to r0 in all directions. If r0 is even, this contains the point (0,0,0),
        otherwise it does not.

        This is quite fast compared to implementations using python list comprehension
        """
        odds = np.array(range(-r0, r0 + 1, 2))
        evens = np.array(range(-r0 + 1, r0 + 1, 2))
        odds_grid = np.array(np.meshgrid(odds, odds, odds)).T.reshape(-1, 3)
        evens_grid = np.array(np.meshgrid(evens, evens, odds)).T.reshape(-1, 3)
        evens_grid2 = np.array(np.meshgrid(evens, odds, evens)).T.reshape(-1, 3)
        odds_grid2 = np.array(np.meshgrid(odds, evens, evens)).T.reshape(-1, 3)
        return np.vstack((odds_grid, evens_grid, evens_grid2, odds_grid2))

    def fcc_dipole_adj(self, layers):
        """
        This function is a hamfisted mess which gets adjacency for up to 6th
        layer spins via dipole connectivity along the [111] direction. This
        DEFINTELY should be generalized for future use. E.g. more code efficient
        generation and

        Args:
            layers (int): number of adjacency layers to consider, a positive integer
        """
        weight_adj = []

        J_vals = [1, 0.32075, 0.19245, 0.13608, 0.125, 0.06415]

        nn_adjacency = [
            np.array([alpha, beta, 0]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        nn_adjacency += [
            np.array([alpha, 0, beta]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        nn_adjacency += [
            np.array([0, alpha, beta]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        weight_adj.append([J_vals[0], nn_adjacency])
        nnn_adjacency = [
            alpha * (np.array([1, 1, 1]) + ei)
            for alpha in (-1, 1)
            for ei in np.identity(3)
        ]
        weight_adj.append([J_vals[1], nnn_adjacency])
        nnnn_adjacency = [
            alpha * (np.array([1, 1, 1]) - 3 * ei)
            for alpha in (-1, 1)
            for ei in np.identity(3)
        ]
        weight_adj.append([J_vals[2], nnnn_adjacency])
        n5_adjacency = [alpha * np.array([1, 1, 1]) for alpha in (-1, 1)]
        weight_adj.append([J_vals[3], n5_adjacency])
        n6_adjacency = [
            2 * np.array([alpha, beta, 0]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        n6_adjacency += [
            2 * np.array([alpha, 0, beta]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        n6_adjacency += [
            2 * np.array([0, alpha, beta]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        weight_adj.append([J_vals[4], n6_adjacency])
        n7_adjacency = [
            2 * alpha * (np.array([1, 1, 1]) - 2 * ei)
            for alpha in (-1, 1)
            for ei in np.identity(3)
        ]
        n7_adjacency += [
            np.array([alpha, -alpha, 2 * beta]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        n7_adjacency += [
            np.array([alpha, 2 * beta, -alpha]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        n7_adjacency += [
            np.array([2 * beta, alpha, -alpha]) for alpha in (-1, 1) for beta in (-1, 1)
        ]
        weight_adj.append([J_vals[5], n7_adjacency])

        return weight_adj
