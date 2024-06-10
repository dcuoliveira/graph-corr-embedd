
#python src/run_simulation1c.py --sample False --graph_name watts_strogatz --n_simulations 5 --n_graphs 5 --n_nodes 100 
python src/run_SDNE3.py --model_name sdne3 --sample False --batch_size 1 --n_hidden 20 --n_layers_enc 1 --n_layers_dec 1 --epochs 10 --dataset_name simulation1c --graph_name watts_strogatz
python src/run_Spectrum.py  --sample False --model_name spectrum --dataset_name simulation1c --graph_name watts_strogatz

