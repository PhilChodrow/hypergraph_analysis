import numpy as np
import networkx as nx
from itertools import combinations
from hypergraph import hypergraph

def simplex_graph(C, weight = 'jaccard'):
    '''
    Nodes are simplices, simplices are connected to each other with weight j if intersection has cardinality j.  
    Nodes have multiplicity -- each node stands for all identical simplices. 
    The number of such simplices is the attribute 'n'.
    '''
    # condensed counter of simplices, represented as tuples
    P = C.C.copy()
    Q = {}
    while len(P) > 0:
        f = P[0]
        k = P.count(f)
        Q.update({tuple(f):k})
        for i in range(k):
            P.remove(f)
    
    G = nx.Graph()
    G.add_nodes_from(Q.keys())
    nx.set_node_attributes(G, Q, 'n')
    edges = []
    
    for f1 in Q:
        for f2 in Q:
            j = jaccard(f1, f2)
            if j > 0:
                edges.append((f1, f2, j))
    
    G.add_weighted_edges_from(edges)
    return G
        
def jaccard(f1, f2):
    return 1.0*len(set(f1) & set(f2)) / len(set(f1) | set(f2))

def line_graph(C, weighted = False, as_hyper = False, multi = True):
    '''
    Compute the line graph corresponding to a given hypergraph. Can be slow when many high-dimensional edges are present. 
    '''
    if not as_hyper:
        if multi:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
        G.add_nodes_from(C.nodes)
        for f in C.C:
            if weighted:
                if len(f) >= 2:
                    G.add_edges_from(combinations(f, 2), weight = 1.0/(len(f) - 1))
            else :
                G.add_edges_from(combinations(f, 2))
        return(G)
    else:
        G = [f for F in C.C for f in combinations(F, 2)]
        return(hypergraph.hypergraph(G, n_nodes = len(C.nodes)))
    
    
