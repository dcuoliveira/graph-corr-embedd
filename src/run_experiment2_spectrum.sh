#!/bin/sh

# test the simulation2a
python3 src/run_simulation2.py --sample False --simulation_name simulation2a --n_simulation 20 --n_nodes 50 --covariance 0

# test the simulation2b
python3 src/run_simulation2.py --sample False --simulation_name simulation2b --n_simulation 20 --n_nodes 50 --covariance 0.6