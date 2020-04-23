import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
from heapq import heappush, heappop

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    pq = []
    for e in G.edges:
        heappush(pq, (-G.edges[e]['weight'], e))
    
    T = G
    while pq:

        e = heappop(pq)[1]
        
        w = G.edges[e]['weight']
        T = T.remove_edge(e)
        
        
    # print(list(G.edges))
    # print(nx.is_dominating_set(G, G.nodes))
    # print(average_pairwise_distance(G))



# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # path = sys.argv[1]
    G = read_input_file("inputs/small-111.in")
    solve(G)
    # assert is_valid_network(G, T)
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'out/test.out')
