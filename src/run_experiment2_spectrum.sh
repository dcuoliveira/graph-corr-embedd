
#!/bin/sh

# test the simulation2a
python3 run_simulation2a.py --sample False --simulation_name simulation2a --graph_types ["erdos_renyi", "random_geometric", "watts_strogatz"] --n_simulation 20 --n_graphs [10,20] --n_nodes 50 --covariance 0

# test the simulation2b
python3 run_simulation2a.py --sample False --simulation_name simulation2a --graph_types ["erdos_renyi", "random_geometric", "watts_strogatz"] --n_simulation 20 --n_graphs [10,20] --n_nodes 50 --covariance 0.6