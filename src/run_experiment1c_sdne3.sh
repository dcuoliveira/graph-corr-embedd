
# Define variables
# ['erdos_renyi', 'k_regular', 'geometric', 'barabasi_albert', 'watts_strogatz']
GRAPH_NAME="watts_strogatz"
SAMPLE="False"

#############
# Run first the graph generation 
#python src/run_simulation1c.py --sample $SAMPLE --graph_name $GRAPH_NAME
#python3 src/run_simulation1c_2.py --graph_name $GRAPH_NAME --sample $SAMPLE # TESTS
##############

##############
# After run the model training for each model
##python src/run_Spectrum.py  --sample $SAMPLE --model_name spectrum --dataset_name simulation1c --graph_name $GRAPH_NAME
#python src/run_SDNE3.py --model_name sdne3 --sample $SAMPLE --batch_size 1 --n_hidden 30 --n_layers_enc 1 --n_layers_dec 1 --epochs 100 --dataset_name simulation1c --graph_name $GRAPH_NAME
#python src/run_SDNE3.py --model_name sdne3 --sample $SAMPLE --batch_size 1 --n_hidden 50 --n_layers_enc 1 --n_layers_dec 1 --epochs 100 --dataset_name simulation1c --graph_name $GRAPH_NAME
#python src/run_SDNE3.py --model_name sdne3 --sample $SAMPLE --batch_size 1 --n_hidden 100 --n_layers_enc 1 --n_layers_dec 1 --epochs 100 --dataset_name simulation1c --graph_name $GRAPH_NAME
##############


python src/run_SAE1.py --model_name sae1 --sample $SAMPLE --batch_size 1 --epochs 100 --dataset_name simulation1c --graph_name $GRAPH_NAME --dropout 0.5 --hidden_size "50,25,50"