import subprocess
from itertools import product
import argparse
import pandas as pd

parser=argparse.ArgumentParser()

parser.add_argument('--pars', 
                    help='if given, pull in pars from given location', 
                    default = None)
args=parser.parse_args()    
args = vars(args)

if args['pars'] is not None:
    print('reading pars from ' + args['pars'])
    pars = pd.read_csv(args['pars'])

    for i in range(len(pars.index)):
        d = pars.iloc[i]
        d = d.to_dict()

        cmd_args = ['data', 'nrounds', 'nsamples',  'sample_after', 'save_state', 'read_state', 'nsteps', 'start_time', 'out_name', 'graph', 'label']
        arg_str = ['--' + arg + ' {' + arg + '}' for arg in cmd_args]
        arg_str = ' '.join(arg_str)
        arg_str = arg_str.format_map(d)

        arg_str += ' --analysis intersection clustering triangles assortativity'
        
        subprocess.call('python3 py/analyze.py ' + arg_str, shell=True)