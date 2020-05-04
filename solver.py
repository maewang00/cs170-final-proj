import networkx as nx
from parse import read_input_file, write_output_file, validate_file
from utils import is_valid_network, average_pairwise_distance_fast
import sys
from heapq import heappush, heappop
import heapq
import copy
import random
import os
import networkx.algorithms.isomorphism as iso

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

    campos_mst = campos_algorithm(G)
    campos_copy = copy.deepcopy(campos_mst)
    for n in campos_copy.nodes:
        if campos_copy.degree(n) == 1:
            campos_mst.remove_node(n)

    bftree = maes_dumb_brute_force(G)
    
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

    result = heappop(costpq)[1]
    #brute_force = maes_dumb_brute_force(G)

    # if average_pairwise_distance_fast(result) <= average_pairwise_distance_fast(MST):
    #     if average_pairwise_distance_fast(bftree) < average_pairwise_distance_fast(result):
    #         print("BRUTEFORCE")
    #         return bftree
    #     else:
    #         print("ORIGIN ALG")
    #         return result
    # else:
    #     print("MST")
    #     return MST

    if average_pairwise_distance_fast(result) <= average_pairwise_distance_fast(MST):
        if average_pairwise_distance_fast(campos_mst) <= average_pairwise_distance_fast(result):
            if average_pairwise_distance_fast(bftree) < average_pairwise_distance_fast(campos_mst):
                print("BRUTEFORCE")
                return bftree
            else:
                print("CAMPOS") 
                return campos_mst
        else: 
            print("ORIGIN ALG") 
            return result
    else:
        print("MST")
        return MST

def maes_dumb_brute_force(G): #;( uses dijkstra's
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

def campos_algorithm(G):
    sum_weights = [0] * G.number_of_nodes() #Sum of weights for each vertex
    max_weight = [0] * G.number_of_nodes() #Maximum weight of adjacent edges for each vertex
    sp = [0] * G.number_of_nodes()
    sp_max = 0
    f = 0 # node with maximum spanning potential
    for n in G.nodes:
        for e in G.edges(n):
            sum_weights[n] = sum_weights[n] + G.edges[e]['weight'] 
            max_weight[n] = max(max_weight[n], G.edges[e]['weight']) 
        sp[n] = 0.2 * G.degree(n) * 0.6 * (G.degree(n)/sum_weights[n]) + 0.2*(1/max_weight[n]) 
        if sp[n] > sp_max:
            sp_max = sp[n]
            f = n

    cf = ['inf'] * G.number_of_nodes() # estimated cost of the path between v and f in T
    for n in G.nodes:
        cf[n] = nx.shortest_path_length(G, source=f, target=n, method='dijkstra')
    
    mst = nx.Graph()
    visited = set([f])
    edges = []
    for u in G.neighbors(f):
        wd_u = 0.9 * G.edges[(u, f)]['weight']  + 0.1 * (cf[u] + G.edges[(u,f)]['weight'])
        jsp_u = (G.degree(f) + G.degree(u)) + (G.degree(f) + G.degree(u))/(sum_weights[u] + sum_weights[f])
        cost = wd_u + (1/jsp_u)
        heappush(edges, (cost, f, u))
    while mst.number_of_edges() < G.number_of_nodes() - 1:
        cost, frm, to = heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add_edge(frm, to, weight=G.edges[(frm, to)]['weight'])
        for to_next in G.neighbors(to):
            wd_n = 0.9 * G.edges[(to, to_next)]['weight']  + 0.1 * (cf[to_next] + G.edges[(to,to_next)]['weight'])
            jsp_n = (G.degree(to) + G.degree(to_next)) + (G.degree(to) + G.degree(to_next))/(sum_weights[to_next] + sum_weights[to])
            cost = wd_n + (1/jsp_n)
            if to_next not in visited:
                heappush(edges, (cost, to, to_next))         
                
    return mst
    

def maes_dumb_brute_force(G): #;( uses dijkstra's
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
  

#void function that updates costPQ
def small_graph_bruteforce_recursive(G, currG, costPQ):
    T = copy.deepcopy(currG)
    #for all edges in the current graph
    for e in T.edges:
        #BASE CASE: continue or don't continue
        if nx.is_connected(currG) and nx.is_dominating_set(G, currG):
            if nx.is_tree(currG):
                heappush(costPQ, (average_pairwise_distance_fast(currG), currG)) #records
                print(costPQ[0])
        else:
            #the current graph is not connected or dominating
            return

        e0 = e[0]
        e1 = e[1]
        w = G.get_edge_data(e0, e1)['weight']

        #remove then recursive
        currG.remove_edge(e[0], e[1])
        if currG.degree(e[1]) == 0:
            currG.remove_node(e[1])
        if currG.degree(e[0]) == 0:
            currG.remove_node(e[0])
        small_graph_bruteforce_recursive(G, currG, costPQ) #recursive call

        #re-add what I just removed
        if e0 not in currG.nodes:
            currG.add_node(e0)
        if e1 not in currG.nodes:
            currG.add_node(e1)
        currG.add_edge(e0, e1, weight=w)


#for small graphs only! Should find the solution indefinitely 
#[UPDATE: DO NOT USE]
def maes_second_dumb_brute_force(G):
    pqcost = []
    small_graph_bruteforce_recursive(G, G, pqcost)
    return heappop(pqcost)[1]


#G is original graph, T is the tree from what we found from our first algorithm, 
#iterations is how many times to randomly find a tree
def maes_randomization_alg(G, T, iterations):
    #min heap with minimal pairwise distance for ALL random trees
    costPQ = []
    costs = []

    for iter in range(iterations):
        #edges in the tree so far
        delete_edges = list(T.edges)
        #how many edges to delete from the start tree initially
        delete_num_edges = random.randint(1, len(delete_edges))
        MODgraph = copy.deepcopy(G)

        #delete "delete_num_edges" VALID edges in total
        while (delete_num_edges > 0 and len(delete_edges) != 0):
            chosen_edge = random.choice(delete_edges) 
            w = MODgraph.get_edge_data(chosen_edge[0], chosen_edge[1])['weight']
            # e0 = chosen_edge[0]
            # e1 = chosen_edge[1]
            MODgraph.remove_edge(chosen_edge[0], chosen_edge[1])
            # if MODgraph.degree(chosen_edge[1]) == 0:
            #     MODgraph.remove_node(chosen_edge[1])
            # if MODgraph.degree(chosen_edge[0]) == 0:
            #     MODgraph.remove_node(chosen_edge[0])
            delete_edges.remove(chosen_edge)

            if not (nx.is_connected(MODgraph) and nx.is_dominating_set(G, MODgraph)):
                #invalid edge,re-add what I just removed
                if chosen_edge[0] not in MODgraph.nodes:
                    MODgraph.add_node(chosen_edge[0])
                if chosen_edge[1] not in MODgraph.nodes:
                    MODgraph.add_node(chosen_edge[1])
                MODgraph.add_edge(chosen_edge[0], chosen_edge[1], weight=w)
            else:
                #found a valid edge
                delete_num_edges -= 1
        #save time to see if NO edges from T can be removed
        if (not nx.is_isomorphic(G, MODgraph)):
            randTree = solve2(MODgraph)
            cost = average_pairwise_distance_fast(randTree)
            if cost not in costs:
                costs.append(cost)
                heappush(costPQ, (cost, randTree))
    
    if len(costPQ) == 0:
        heappush(costPQ, (average_pairwise_distance_fast(T), T))
    return heappop(costPQ)[1]




def solve2(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    pq = [] #min heap for Kruskals edge popping
    for e in G.edges:
        heappush(pq, (-G.edges[e]['weight'], e))
    T = copy.deepcopy(G)
    costpq = [] #min heap with minimal pairwise distance with its tree

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

    return heappop(costpq)[1]




def is_spanning(G, node):
    return (G.degree(node) >= (G.number_of_nodes() - 1))

def makeAllOutputFiles():
    for file in os.listdir("inputs"):
        if file.endswith(".in"):
            print(os.path.join("inputs", file)) #input file
            input_path = os.path.join("inputs", file)
            G = read_input_file(input_path)
            try:
                T = solve(G)
            except:
                print("ERRORED OUT. CONTINUE ANYWAY")
                T = G
            assert is_valid_network(G, T)

            #randomization optimization
            if len(T) > 2:
                print("Trying randomization to find better result..")
                try:
                    betterT = maes_randomization_alg(G, T, 100) #50 iterations of randomness
                except:
                    print("ERRORED OUT. CONTINUE ANYWAY")
                    betterT = G
                assert is_valid_network(G, betterT)

                if average_pairwise_distance_fast(betterT) < average_pairwise_distance_fast(T):
                    print("BETTER TREE FOUND.")
                    T = betterT
                else:
                    print("No improvements.")
                    #nothing happens




            #print("Average pairwise distance: {}".format(average_pairwise_distance_fast(T)))
            outname = os.path.splitext(file)[0]+'.out'
            output_path = os.path.join("outputs", outname)
            print(output_path + "\n")
            write_output_file(T, output_path)
            assert validate_file(output_path) == True

def validateAllFiles():
    for file in os.listdir("outputs"):
        output_path = os.path.join("outputs", file)
        if (validate_file(output_path) == False):
            print(output_path + " INVALIDATED.")
        

makeAllOutputFiles()






# gr = read_input_file('inputs/small-6.in')
# s = maes_second_dumb_brute_force(gr)
# print(average_pairwise_distance_fast(s))
#write_output_file(s, 'outputs/small-6.out')
# print(is_valid_network(gr,s))







#TRASH CODE
# #use brute force for small graphs
# if file.startswith("small") and len(T) > 2:
#     print("Trying brute forcing on SMALL file: " + os.path.join("inputs", file)) #input file
#     BRUTE_TREE = maes_second_dumb_brute_force(G)
#     if average_pairwise_distance_fast(BRUTE_TREE) <= average_pairwise_distance_fast(T):
#         print("Small brute-force alg WINS.")
#         T = BRUTE_TREE
#     else:
#         print("Solver alg WINS.")
#         #nothing happens






        
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
