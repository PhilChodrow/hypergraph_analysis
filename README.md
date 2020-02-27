# Analyzing Polyadic Data with Hypergraph Configuration Models

This repository accompanies the paper "Configuration Models of Random Hypergraphs" by Phil Chodrow, providing the software used to perform analyses in that paper. It is not intended for wide use, but rather only for those aiming to reproduce these results or perform extremely similar analyses. If you'd like to do your own analyses with hypergraph configuration models, please see [`PhilChodrow/hypergraph`](https://github.com/PhilChodrow/hypergraph). 

# Installation

In order to run the software in this repository: 

1. Clone or download the repository to your computer. 
2. Clone or download [`PhilChodrow/hypergraph`](https://github.com/PhilChodrow/hypergraph) **in the .py folder of this repository.**
3. Install the `pandas`, `scipy`, `networkx`, and `googledrivedownloader` packages for python. 

# Usage

## Acquire Data

The paper uses data kindly hosted by Austin Benson on his [website](https://www.cs.cornell.edu/~arb/data/index.html). The script `py/get_data.py` can be used to obtain data from this website. Edit the definition of the `files` dict to include the name of the data set you wish to download and its viewable Google Drive link. Then, run 

```
python py/get_data.py
```

A `data` directory will be created, and the data sets you requested will appear as subdirectories. 

## Perform Analysis

The script `py/control_analysis.py` is used to perform sampling and analysis. It is controlled by a parameter file which determines which data sets to analyze and in what fashion. The syntax is: 

```
python py/control_analysis.py --pars pars.csv
```

In order to minimize compute time, the repo provides a `pars_enron.csv` file which will perform a brief analysis on the `email-enron` data set only. The complete `pars.csv` file enumerates the parameters used in the paper.   