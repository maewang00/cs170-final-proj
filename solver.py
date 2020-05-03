import networkx as nx
from parse import read_input_file, write_output_file, validate_file
from utils import is_valid_network, average_pairwise_distance_fast
import sys
from heapq import heappush, heappop
import copy
import random
import os
def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    #For input graphs of 1 vertex (with possible trick self-loops)
    if G.number_of_nodes() == 1:
        T = nx.Graph()
        T.add_node(0)
        return T

    #For input graphs of 2 vertices and an edge between them
    if G.number_of_nodes() == 2:
        #print("HI")
        e = G.edges[list(G.edges)[0]]['weight'] #list(G.edges)[0]
        G.edges[list(G.edges)[0]]['weight'] = 0.000
        return G

    #For input graphs where 1 vertex spans all other vertices
    spanner_nodes = [node for node in G if is_spanning(G, node)]
    #print(spanner_nodes)
    if (len(spanner_nodes) > 0):
        #print("YO")
        T = nx.Graph()
        if (spanner_nodes[0] == 1):
            T.add_weighted_edges_from([(spanner_nodes[0],2,0)])
        else:
            T.add_weighted_edges_from([(spanner_nodes[0],1,0)])
        return T

    pq = [] #min heap for Kruskals edge popping
    for e in G.edges:
        heappush(pq, (-G.edges[e]['weight'], e))
    T = copy.deepcopy(G)
    costpq = [] #min heap with minimal pairwise distance with its tree
    MST = nx.minimum_spanning_tree(G)
    MST_copy = copy.deepcopy(MST)
    for n in MST_copy.nodes:
        if MST_copy.degree(n) == 1:
            MST.remove_node(n)
    # heappush(costpq, (average_pairwise_distance_fast(MST), MST))
    
    while pq:
        if (T.number_of_nodes() == 2):
            break
        node = heappop(pq)
        e = node[1]
        # print(e)
        w = node[0] * -1
        T.remove_edge(e[0], e[1])
        if T.degree(e[1]) == 0:
            T.remove_node(e[1])
        if T.degree(e[0]) == 0:
            T.remove_node(e[0])
        if nx.is_connected(T) and nx.is_dominating_set(G, T):
            if nx.is_tree(T):
                heappush(costpq, (average_pairwise_distance_fast(T), T))
                # cost = average_pairwise_distance_fast(T)
        else:
            T.add_edge(e[0], e[1], weight=w)

    # #randomize version for 10 iterations
    # iterations = 10
    # while (iterations):
    #     #random.shuffle(G.edges)
    #     for e in G.edges:
    #         heappush(pq, (-G.edges[e]['weight'], e))
    #     _T = copy.deepcopy(G)

    #     while pq:
    #         if (_T.number_of_nodes() == 2):
    #             break
    #         edge = heappop(pq)
    #         e = edge[1]
    #         w = edge[0] * -1
    #         _T.remove_edge(e[0], e[1])
    #         if _T.degree(e[1]) == 0:
    #             _T.remove_node(e[1])
    #         if _T.degree(e[0]) == 0:
    #             _T.remove_node(e[0])
    #         if nx.is_connected(_T) and nx.is_dominating_set(G, _T):
    #             if nx.is_tree(_T):
    #                 heappush(costpq, (average_pairwise_distance_fast(_T), _T))
    #                 # cost = average_pairwise_distance_fast(T)
    #         else:
    #             _T.add_edge(e[0], e[1], weight=w)
    #     iterations -= 1
    result = heappop(costpq)[1]
    brute_force = maes_dumbass_brute_force(G)

    if average_pairwise_distance_fast(result) <= average_pairwise_distance_fast(MST):
        if average_pairwise_distance_fast(result) <= average_pairwise_distance_fast(brute_force):
            print("original alg WINS.")
            return result
        else:
            print("brute force alg WINS.")
            return brute_force
    else:
        if average_pairwise_distance_fast(MST) <= average_pairwise_distance_fast(brute_force):
            print("MST WINS.")
            return MST
        else:
            print("brute force alg WINS.")
            return brute_force
    

def maes_dumbass_brute_force(G): #;( uses dijkstra's
    #min heap with minimal pairwise distance with its tree
    costpq = []
    #list to keep track of all costs found so far
    mincosts = []
    #for all vertices in the graph, make it a source node for dijkstra's
    for source_node in range(0, G.number_of_nodes()):
        shortest_path = nx.shortest_path(G, source = source_node)
        #convert shortest path dictionary into a tree and put into the min heap
        Temp_Tree = nx.Graph()
        edge_list = []
        #iterate through the entire dictionary where: 
        #key is target node from the dummy node
        #value is list where index 0 is the dummy node and the last is the key (target node)
        for key, list_path in shortest_path.items():
            if (len(list_path) > 1):
                for i in range(0, len(list_path) - 1):
                    edge_weight = G.get_edge_data(list_path[i], list_path[i+1])['weight']
                    edge = (list_path[i], list_path[i+1], edge_weight)
                    #edge doesn't exist yet in the graph
                    if (edge not in edge_list):
                        edge_list.append(edge)
        Temp_Tree.add_weighted_edges_from(edge_list)

        #trim down the tree
        tree_copy = copy.deepcopy(Temp_Tree)
        for node in tree_copy.nodes:
            if tree_copy.degree(node) == 1 or tree_copy.degree(node) == 0:
                Temp_Tree.remove_node(node)

        curr_cost = average_pairwise_distance_fast(Temp_Tree)
        if curr_cost not in mincosts:
            heappush(costpq, (curr_cost, Temp_Tree))
            mincosts.append(curr_cost)

    return heappop(costpq)[1]






  #T = copy.deepcopy(G)
  # original_vertices = G.number_of_nodes()
  #make dummy node and connect to all other vertices with edge weights 0
  # T.add_node(original_vertices)
  # dummy_node_id = G.number_of_nodes()
  # for node in range(0, original_vertices):
  #     T.add_weighted_edges_from([(G.number_of_nodes(),node,0)])
  #dictionary where: 
  #key is target node from the dummy node
  #value is list where index 0 is the dummy node and the last is the key (target node)
  # shortest_path = nx.shortest_path(T, source = dummy_node_id)
  #trim down leaf nodes form the shortest path tree
  




def is_spanning(G, node):
    return (G.degree(node) >= (G.number_of_nodes() - 1))

def makeAllOutputFiles():
    for file in os.listdir("inputs"):
        if file.endswith(".in"):
            print(os.path.join("inputs", file)) #input file
            input_path = os.path.join("inputs", file)
            G = read_input_file(input_path)
            T = solve(G)
            assert is_valid_network(G, T)
            #print("Average pairwise distance: {}".format(average_pairwise_distance_fast(T)))
            outname = os.path.splitext(file)[0]+'.out'
            output_path = os.path.join("outputs", outname)
            print(output_path + "\n")
            write_output_file(T, output_path)
            assert validate_file(output_path) == True;

def validateAllFiles():
    for file in os.listdir("outputs"):
        output_path = os.path.join("outputs", file)
        if (validate_file(output_path) == False):
            print(output_path + " INVALIDATED.")
        


makeAllOutputFiles()
# gr = read_input_file('inputs/small-254.in')
# s = solve(gr)
# print(average_pairwise_distance_fast(s))
# write_output_file(s, 'outputs/small-254.out')
# print(is_valid_network(gr,s))

        
    # print(list(G.edges))
    # print(nx.is_dominating_set(G, G.nodes))
    # print(average_pairwise_distance(G))


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file("inputs/small-111.in")
#     solve(G)
#     assert is_valid_network(G, T)
#     print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
#     write_output_file(T, 'output/test.out')
