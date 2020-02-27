import argparse, sys, os
import read
import pandas as pd
import numpy as np
from itertools import product, combinations
from measures import sample_intersection
from transform import line_graph
import pickle
import pathlib
import networkx as nx


# ---------------------------------------------------------------
# CONTROL STRUCTURE
# ---------------------------------------------------------------

def get_args():

    parser=argparse.ArgumentParser()

    parser.add_argument('--data', 
                        help='data set to analyze')
    
    parser.add_argument('--nrounds', 
                        help='number of total sampling rounds', 
                        type = int)
    
    parser.add_argument('--nsteps', 
                        help='Number of steps per Monte Carlo run', 
                        type = int)
    
    parser.add_argument('--nsamples', 
                        help='number of samples to pull for assortativity and intersection computations',
                        type = int)
    
    parser.add_argument('--label', 
                        help='stub, vertex labeling, or empirical', type = str,
                        default = 'empirical')
    
    parser.add_argument('--analysis', 
                        type = str,
                        help='intersection, assortativity, or both', nargs = '+', 
                        default = ['intersection', 'assortativity', 'clustering', 'triangles'])
    
    parser.add_argument('--graph', 
                        help='hypergraph or projected graph', 
                        default = 'hypergraph')
    
    parser.add_argument('--sample_after',
                       help='sample after this round', 
                       default = 0, 
                       type = int)
    
    parser.add_argument('--save_state',
                        help='if true, save a copy of the current graph state in readable format',
                        default = False,
                        type = bool)
    
    parser.add_argument('--read_state', 
                        help='if true, read state from previously saved',
                        default=False,
                        type = bool)
    
    parser.add_argument('--start_time', 
                        help='date at which to begin analysis',
                        default=0,
                        type = float)
    
    parser.add_argument('--out_name',
                        help ='custom throughput directory name if desired',
                        default = None,
                        type = str)
    
    args=parser.parse_args()
    
    args = vars(args)
            
    return(args)

def setup_dirs(data):
    
    if args['out_name'] is None:
        out_dir = data
    else:
        out_dir = args['out_name']
    
    path              = 'data/' + data
    throughput_path   = 'throughput/' + out_dir + '/'  
    intersection_path = throughput_path
    assortative_path  = throughput_path
    save_dir          = throughput_path + 'snapshots/'

    for dir in [throughput_path, intersection_path, assortative_path, save_dir]:
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True) 
    return(path, throughput_path, intersection_path, assortative_path, save_dir)

# ---------------------------------------------------------------
# DESCRIPTIVES
# ---------------------------------------------------------------

def print_descriptives(C, throughput_path):

    deg = C.node_degrees()    
    K = C.edge_dimensions()
    
    n = len(deg)
    m = len(K)
    
    edge_dimension_df = pd.DataFrame({
        'K' : K
    })
    degree_df = pd.DataFrame({
        'D' : deg
    })
    
    edge_dimension_df.to_csv(throughput_path + 'edge_dimension.csv', index = False)
    degree_df.to_csv(throughput_path + 'degree.csv', index = False)

# ---------------------------------------------------------------
# INTERSECTIONS
# ---------------------------------------------------------------

def analyze_intersection(C, method, n_samp):
    df = sample_intersection(C, n_samp)
    df['label'] = method
    df['graph'] = args['graph']
    df['i'] = C.MH_rounds
    df['n_steps'] = C.MH_steps
    df['data'] = args['data']
    df['a'] = C.acceptance_rate
    
    df_path = intersection_path + 'intersection.csv'
    
    path_exists = pathlib.Path(df_path).exists()
    
    with open(df_path, 'a+') as f:
        df.to_csv(f, header = not path_exists, index = False)
# ---------------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------------

def clustering(C):
    G = line_graph(C, weighted = False)
    G_ = nx.Graph(G)
    return nx.average_clustering(G_)

def analyze_clustering(C, method):
    cluster_path = throughput_path + 'clustering.csv'
    if not pathlib.Path(cluster_path).exists():
        with open(cluster_path, 'a+') as f:
            f.write('data, label, graph, i, n_steps, a, C')
    clus = clustering(C)
    with open(cluster_path, 'a+') as f:
        f.write('\n' + args['data'] + ', ' + method + ', ' + args['graph'] + ',' + str(C.MH_rounds) + ', ' + str(C.MH_steps) + ', ' + str(round(C.acceptance_rate, 5)) + ',' + str(round(clus, 5)))
    
    
# ---------------------------------------------------------------
# TRIANGLES
# ---------------------------------------------------------------
def count_triangles(C):
    G = line_graph(C, as_hyper=False)
    G_ = nx.Graph(G)
    triangles = nx.triangles(G_)
    n_triangles = sum(triangles.values())/3
    return(n_triangles)

def count_filled_triangles(C):
    container = set()
    for f in C.C:
        for t in combinations(f, 3):
            container.add(tuple(sorted(t)))
    return(len(container))

def analyze_triangles(C, method):
    triangle_path = throughput_path + 'triangles.csv'
    if not pathlib.Path(triangle_path).exists():
        with open(triangle_path, 'a+') as f:
            f.write('data, label, graph, i, n_steps, a, n_triangles, n_filled')
    n_triangles = count_triangles(C)
    n_filled_triangles = count_filled_triangles(C)
    with open(triangle_path, 'a+') as f:
        f.write('\n' + args['data']+ ', ' + method  + ', ' + args['graph'] + ', ' + str(C.MH_rounds) + ',' + str(C.MH_steps) + ',' + str(round(C.acceptance_rate, 5)) + ',' + str(n_triangles) + ',' + str(n_filled_triangles))
    
    
# ---------------------------------------------------------------
# ASSORTATIVITY
# ---------------------------------------------------------------

def analyze_assortativity(C_, method, n_samples, all_choices = True):
#         cors = ['spearman', 'pearson']    
    cors = ['spearman']
    if all_choices:
        df = pd.DataFrame(list(product(cors, ['top_2', 'top_bottom', 'uniform'])), columns = ['cor', 'choice'])
    else:
        df = pd.DataFrame(cors, columns = ['cor'])
        df['choice'] = 'NA'
    
    def f(x):
        return(C_.assortativity(n_samples, x['choice'], x['cor']))
        
    df['coef'] = df.apply(f, axis = 1)
    df['n_samples'] = n_samples
    df['label'] = method
    df['graph'] = args['graph']
    df['i'] = C.MH_rounds
    df['n_steps'] = C.MH_steps
    df['data'] = args['data']
    df['a'] = C.acceptance_rate  
    
    df_path = assortative_path + 'assortativity.csv'
    
    path_exists = pathlib.Path(df_path).exists()
    
    with open(df_path, 'a+') as f:
        df.to_csv(f, header = not path_exists, index = False)
    
    
# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------

if __name__ == "__main__":
     
    # Get the arguments
    args = get_args()
        
    print('analyzing data set: '  + args['data'])
    
    # setup directory structure
    path, throughput_path, intersection_path, assortative_path, save_dir = setup_dirs(args['data'])
    
    # read the data
    
    snapshot_path = save_dir + args['label'] + '_' + args['graph'] + '.p'
    
    if args['label'] == 'empirical':
        D = read.read_data(path, labels = False)
        print('data starts at t = ' + str(D['times'].time.min()))
        print('truncating to ' + str(args['start_time']))
        D = read.time_filter(D, labels = False, t = args['start_time'])
        C = read.as_hypergraph(D, node_labels = False)
        state_read = False
    elif args['read_state']:
        print(snapshot_path)
        try:
            C = pickle.load(open(snapshot_path, "rb" ))
            print('reading *.p file')
            state_read = True
        except Exception:
            print('no *.p file located, reading original data')
            D = read.read_data(path, labels = False)
            print('data starts at t = ' + str(D['times'].time.max()))
            print('truncating to ' + str(args['start_time']))
            D = read.time_filter(D, labels = False, t = args['start_time'])
            C = read.as_hypergraph(D, node_labels = False)
            state_read = False
    
    # temporal threshold
    
    # print out basic descriptives
    if args['label'] == 'empirical':
        print_descriptives(C, throughput_path)
    
    
    # project if required
    
    if args['graph'] == 'projected':
        if not state_read:
            print('Projecting graph')
            C = line_graph(C, as_hyper = True) # projected line graph, includes multiple edges. 
            print('projected graph has ' + str(len(C.C)) + ' edges.')
        
    

    if args['nsteps'] is None:
        K = C.edge_dimensions()
        kappa_1 = K.mean()
        m = len(K)
        t_mix = int(round(m*np.log(m)*kappa_1))
        n_steps = int(20*t_mix / int(args['nrounds']))
        print('nsteps not provided, using heuristic with ' + str(n_steps) + ' steps')
    else:
        n_steps = args['nsteps']
    

    def do_analysis(method):
        is_hypergraph = (args['graph'] == 'hypergraph')
        if is_hypergraph:
            method_name = method
        else:
            method_name = method
        
        if method == 'empirical':
            n_samp = args['nsamples']*100
        else:
            n_samp = args['nsamples']
        
        if 'assortativity' in args['analysis']:
            print('analyzing assortativity')
            analyze_assortativity(C, method_name, int(n_samp), 
                                  all_choices = is_hypergraph)
        if ('intersection' in args['analysis']) & (is_hypergraph):
            print('analyzing intersection')
            analyze_intersection(C, method_name, int(n_samp))
        
        if ('clustering' in args['analysis']):
            print('analyzing clustering')
            analyze_clustering(C, method_name)
            
        if ('triangles' in args['analysis']) & (is_hypergraph):
            print('analyzing triangles')
            analyze_triangles(C, method_name)
            
    label = args['label']
    
    if label == 'empirical':
        i = 0
        print('Performing empirical analysis')
        do_analysis('empirical')
        
    else:
        m = len(C.C)
        n_warmup_steps = 10*round(m*np.log(m))
        n_warmup_steps = 10
        print('stub-matching warm-start with ' + str(n_warmup_steps) + ' steps.')
        a = C.MH(n_warmup_steps, verbose = False, label = 'stub', message = False)
        
        i = C.MH_rounds
        for i in np.arange(i, i + int(args['nrounds'])):
            print(path + ' || ' + label + '-labeled with ' + str(int(n_steps)) + ' steps: round ' + str(i))
            a = C.MH(n_steps, verbose = False, label = label, n_clash = 1, message = True)
            if i > args['sample_after']:
                do_analysis(label)

        if args['save_state']:
            with open(snapshot_path, "wb" ) as handle:
                pickle.dump(C, handle)