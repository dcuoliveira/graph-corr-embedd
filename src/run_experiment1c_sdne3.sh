
#python src/run_simulation1c.py --sample False --graph_name geometric --n_simulations 5 --n_graphs 5 --n_nodes 100 
python src/run_simulation1c.py --sample False --graph_name geometric 

python src/run_Spectrum.py  --sample False --model_name spectrum --dataset_name simulation1c --graph_name geometric
python src/run_SDNE3.py --model_name sdne3 --sample False --batch_size 1 --n_hidden 30 --n_layers_enc 1 --n_layers_dec 1 --epochs 100 --dataset_name simulation1c --graph_name geometric
python src/run_SDNE3.py --model_name sdne3 --sample False --batch_size 1 --n_hidden 50 --n_layers_enc 1 --n_layers_dec 1 --epochs 100 --dataset_name simulation1c --graph_name geometric
python src/run_SDNE3.py --model_name sdne3 --sample False --batch_size 1 --n_hidden 100 --n_layers_enc 1 --n_layers_dec 1 --epochs 100 --dataset_name simulation1c --graph_name geometric 


# TODO: Analyse result for  geometric graph






