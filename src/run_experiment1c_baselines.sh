# python run_DeltaCon.py --model_name deltacon --sample False --load_preprocessed False --dataset_name simulation1c --graph_name erdos_renyi
# python run_DeltaCon.py --model_name deltacon --sample False --load_preprocessed False --dataset_name simulation1c --graph_name watts_strogatz
# python run_Frobenius.py --model_name frobenius --sample False --load_preprocessed True --dataset_name simulation1c --graph_name erdos_renyi
# python run_Frobenius.py --model_name frobenius --sample False --load_preprocessed True --dataset_name simulation1c --graph_name watts_strogatz
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name erdos_renyi --k 1
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name watts_strogatz --k 1
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name erdos_renyi --k 2
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name watts_strogatz --k 2
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name erdos_renyi --k 3
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name watts_strogatz --k 3
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name erdos_renyi --k 4
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name watts_strogatz --k 4
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name erdos_renyi --k 5
python run_Procustes.py --model_name procustes --sample False --load_preprocessed True --dataset_name simulation1c --graph_name watts_strogatz --k 5