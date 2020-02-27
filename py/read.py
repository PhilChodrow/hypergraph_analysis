import pandas as pd
import numpy as np
from hypergraph import hypergraph

COL_NAMES = {'node-labels' : ['id', 'label'],
                 'nverts'      : ['nverts'],
                 'simplices'   : ['id'],
                 'times'       : ['time']}

def pathify(path, file_type):
    split_path = path.split('/')
    directory = '/'.join(split_path[:-1])
    name = split_path[-1]
    return directory + '/' + name + '/' + name + '-' + file_type + '.txt'

def read_file(path, file_type):
    return pd.read_table(pathify(path, file_type),
                         delimiter = ' ',
                         names = COL_NAMES[file_type])

def read_data(path, labels = False):
    names = list(COL_NAMES.keys())
    if not labels:
        names.remove('node-labels')
    D = {l : read_file(path, l) for l in names}
    return D

def clean_nodes(D, labels = True):
    
    # figure out which nodes appear in any simplices
    count_df = D['simplices'].groupby(['id']).aggregate(np.sum)
    
    # impose new labels
    count_df = count_df.reset_index().reset_index()
    count_df = count_df.rename(index = str, columns = {'index' : 'id', 'id' : 'label_id'})
    count_df = count_df.set_index('label_id')
    
    # join to node-labels data, and add as new ids
    def relabel(df):
        df = df.rename(index = str, columns = {'id': 'label_id'})
        df = df.reset_index()
        df = pd.merge_ordered(df, count_df, how= 'left', on = 'label_id')
        df['index'] = df['index'].astype(int)
        df = df.sort_values('index').reset_index()
        df = df.drop(['index', 'label_id', 'level_0'], axis = 1)
        df = df[np.isfinite(df.id)]
        df['id'] = df['id'].astype(int)
        return df
    
    
    if labels:
        D['node-labels'] = relabel(D['node-labels'])
    D['simplices']   = relabel(D['simplices'])
    
    return D

def time_filter(D, labels = True, t = 0):
    expanded_times = np.repeat(D['times']['time'], repeats = D['nverts']['nverts'])
    expanded_times = expanded_times.reset_index().drop(['index'], axis = 1)
    D['nverts'] = D['nverts'][D['times'].time >= t]
    D['times'] = D['times'][D['times'].time >= t]
    D['simplices'] = D['simplices'][expanded_times.time >= t]
    D = clean_nodes(D, labels)
    return(D)

def as_hypergraph(D, **kwargs):
    C = []
    ix = 0
    D['simplices'] = D['simplices'].reset_index()
    
    nverts = np.array(D['nverts'].nverts)
    nverts = np.cumsum(nverts)
    id_vec = np.array(D['simplices'].id)
    
    C = np.split(id_vec, nverts)
    C = [list(v) for v in C]
    
    return hypergraph.hypergraph(C, **kwargs)