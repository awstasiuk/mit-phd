import rustworkx as rx


class Graph:
    def __init__(self):
        self.graph = rx.PyGraph()

    def add_node(self, node):
        self.graph.add_node(node)

    def add_edge(self, u, v, weight=None):
        self.graph.add_edge(u, v, weight)

    def get_nodes(self):
        return self.graph.nodes()

    def get_edges(self):
        return self.graph.edges()
