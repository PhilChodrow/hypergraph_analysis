import numpy as np
import pandas as pd
from collections import Counter
def square_dist(C): 
    
    degs = C.node_degrees(by_dimension = True) 
    degs = degs / degs.sum()
    prod = 1.0*np.outer(degs.sum(axis = 0).T,degs.sum(axis = 1)) 
    return np.power(degs.T - prod, 2).sum()

def KL_dist(C):
    degs = C.node_degrees(by_dimension = True) 
    degs = degs / degs.sum()
    prod = 1.0*np.outer(degs.sum(axis = 0).T,degs.sum(axis = 1)) 
    return np.nansum(degs * np.log(degs / prod.T))
    

# def sample_intersection(C, n_samples):
    
#     v  = np.zeros(n_samples)
#     k1 = np.zeros(n_samples)
#     k2 = np.zeros(n_samples)
    
#     m = len(C.C)
    
#     k_rand = 2*n_samples
#     IJ = np.random.randint(0,m, size = (k_rand, 2))
#     k_ = 0
#     i = 0
#     while i < n_samples:
#         m1, m2 = IJ[i,0], IJ[i,1]
#         if m1 == m2:
#             pass
#         else:
#             f1 = C.C[m1]
#             f2 = C.C[m2]
#             v[i] = len(set(f1) & set(f2))
#             k1[i] = len(f1)
#             k2[i] = len(f2)
#             i += 1
#         k_ += 1
#         if k_ > k_rand:
#             IJ = np.random.randint(m, size = (k_rand, 2))
                 
#     df = pd.DataFrame({'k_1': k1, 'k_2' : k2, 'j' : v})
#     df = df.groupby(['j', 'k_1', 'k_2']).size().reset_index(name='n')
#     for col in ['j', 'k_1', 'k_2', 'n']:
#         df[col] = df[col].astype(int)
#     return df

def sample_intersection(C, n_samples):
    
    m = len(C.C)
    
    k_rand = 20000
    IJ = np.random.randint(0, m, size = (k_rand+1, 2))
    k_ = 0
    i = 0
    
    counts = Counter()
    
    while i < n_samples:
#         m1, m2 = np.random.randint(0, m, size = 2)
        m1, m2 = IJ[k_,0], IJ[k_,1]
        if m1 == m2:
            pass
        else:
            f1 = C.C[m1]
            f2 = C.C[m2]
            counts[(len(f1), len(f2), len(set(f1) & set(f2)))] += 1
            i += 1
        k_ += 1
        if k_ >= k_rand:
            IJ = np.random.randint(m, size = (k_rand, 2))
            k_ = 0
            
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df['k_1'] = df['index'].apply(lambda x: x[0])
    df['k_2'] = df['index'].apply(lambda x: x[1])
    df['j'] = df['index'].apply(lambda x: x[2])
    df = df.rename({0 : 'n'}, axis = 1)
    df = df.drop('index', axis = 1)
    
    return df




# issue here: what if no neighbors? 

def local_clustering(C, u, n_samples):
    
    # obtain w and v, distinct neighbors of u. 
    matched = False
    while not matched:
        v = get_random_neighbor(u)
        w = v
        while w == v:
            w = get_random_neighbor(u)
    
        
    def get_random_neighbor(u):
        matched = False
        while not matched:
            v = np.random.choice(range(C.m))
            if v != u:
                if len(set(C.C[u]) & set(C.C[v])) > 0:
                    matched = True
        return v

def neighbor_degree(C, n_samples):
    D = C.node_degrees()
    D_1 = []
    D_2 = []
    
    set_edges = [set(f) for f in C.C]
    
    while len(D_1) < n_samples:
        u = C.nodes[np.random.randint(len(C.nodes))]
        d_1 = D[u]
        edges = [f for f in set_edges if u in f]
        if d_1 < 1:
            d_2 = np.nan()
            D_1.append(d_1)
            D_2.append(d_2)
        else:
            edge = edges[np.random.randint(0, len(edges))]
            if len(edge) == 1:
                pass
            else:
                v = np.random.choice([n for n in edge if n != u])
                d_2 = D[v]
                D_1.append(d_1)
                D_2.append(d_2)
                
    return(D_1, D_2)

def neighbor_multimoment(C, n_samples):
    D = C.node_degrees()
    mu_1 = D.mean()
    vec = []
#     pr = []
#     K = []
    n = 0
    while n < n_samples:
        edge = np.random.choice(C.C)
        if len(edge) == 1:
            continue
#         pr.append(np.log(D[edge,]).sum())
#         K.append(len(edge))
#         top    =  (np.prod(D[edge,]) - mu_1**(len(edge)))
#         bottom = (D**(len(edge))).mean() - mu_1**(len(edge))
#         vec.append((1.0*top) / bottom)
        new = np.prod(D[edge,]**(1.0/len(edge)))
        vec.append(new)
#         vec.append(np.log((np.prod(D[edge,]) /mu_1**(len(edge)))))
        n += 1
    return(np.array(vec))


def sample_network_nodes(G, n_samples):
    edge = list(G.edges(data = False))
    m = len(edge)
    ix = np.random.randint(m, size = n_samples)
    return([edge[i] for i in ix])
        
    
    
def network_assortativity(G, n_samples = 100, method = 'pearson'):
    nodes = np.array(sample_network_nodes(G, n_samples))
    deg = dict(G.degree)
    arr = np.vectorize(lambda x: deg[x])(nodes)
#     D = np.array([deg[i] for i in range(len(deg))])
#     arr = 
    
#     D[nodes]
    
    if method == 'spearman':
        order = np.argsort(arr, axis = 0)
        arr = np.argsort(order, axis = 0)
    elif method == 'pearson':
        arr = arr - 1
    
    return(np.corrcoef(arr.T))[0,1]
    
#     for n in range(n_samples):
        


