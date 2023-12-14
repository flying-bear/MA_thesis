import networkx as nx
import numpy as np

from collections import Counter
from typing import List, Optional, Dict


# https://github.com/facuzeta/speechgraph/blob/master/speechgraph/speechgraph.py

class _graphStatistics:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def statistics(self) -> Dict[str, float]:
        res = {}
        graph = self.graph
        res['number_of_nodes'] = graph.number_of_nodes()
        res['number_of_edges'] = graph.number_of_edges()
        res['PE'] = (np.array(list(Counter(graph.edges()).values())) > 1).sum()
        res['LCC'] = len(max(nx.weakly_connected_components(graph), key=len))
        res['LSC'] = len(max(nx.strongly_connected_components(graph), key=len))

        degrees = [v for k, v in graph.degree()]
        res['degree_average'] = np.mean(degrees)
        res['degree_std'] = np.std(degrees)

        adj_matrix = nx.adjacency_matrix(graph).toarray()
        adj_matrix2 = np.dot(adj_matrix, adj_matrix)
        adj_matrix3 = np.dot(adj_matrix2, adj_matrix)

        res['L1'] = np.trace(adj_matrix)
        res['L2'] = np.trace(adj_matrix2)
        res['L3'] = np.trace(adj_matrix3)

        return res


class NaiveGraph:
    @staticmethod
    def _text2graph(words: List[str]) -> nx.MultiDiGraph:
        gr = nx.MultiDiGraph()
        gr.add_edges_from(zip(words[:-1], words[1:]))
        return gr

    def analyzeText(self, words: List[str]) -> Dict[str, float]:
        dgr = self._text2graph(words)
        return _graphStatistics(dgr).statistics()


def moving_graph_statistics(words: List[str], w: Optional[int] = 15) -> Dict[str, float]:
    if len(words) <= w:
        return NaiveGraph().analyzeText(words)
    else:
        graph_stats = {}
        for i in range(len(words) - w):
            current_stats = NaiveGraph().analyzeText(words[i:i + w])
            graph_stats = {key: graph_stats.get(key, []) + [value] for key, value in current_stats.items()}
        return {key: np.mean(value) for key, value in graph_stats.items()}
