import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance_fast
import sys
from heapq import heappush, heappop
import copy
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
    
    T = copy.deepcopy(G)
    # print(T == G)
    print(type(T))
    costpq = []
    
    while pq:
        node = heappop(pq)
        e = node[1]
        # print(e)
        w = node[0] * -1
        T.remove_edge(e[0], e[1])
        if T.degree(e[1]) == 0:
            T.remove_node(e[1])
        if T.degree(e[0]) == 0:
            T.remove_node(e[0])
        # print("CONNECT")
        # print()
        if nx.is_connected(T) and nx.is_dominating_set(G, T):
            # print("SDFSDF")
            if nx.is_tree(T):
                heappush(costpq, (average_pairwise_distance_fast(T), T))
                cost = average_pairwise_distance_fast(T)
        else:
            T.add_edge(e[0], e[1], weight=w)
    # return 0
    return heappop(costpq)[1]
    

# gr = read_input_file('inputs/small-4.in')
# s = solve(gr)
# print(average_pairwise_distance_fast(s))
# print(is_valid_network(gr,s))
        
    # print(list(G.edges))
    # print(nx.is_dominating_set(G, G.nodes))
    # print(average_pairwise_distance(G))



# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     # assert len(sys.argv) == 2
#     # path = sys.argv[1]
#     G = read_input_file("inputs/small-111.in")
#     solve(G)
    # assert is_valid_network(G, T)
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'out/test.out')
