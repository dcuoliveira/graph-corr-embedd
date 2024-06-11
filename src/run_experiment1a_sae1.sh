
python run_SAE1.py --dataset_name simulation1a --model_name sae --sample False --input_size 100 --dropout 0.5 --epochs 10 --learning_rate 0.001 --sparsity_penalty 1e-4 --hidden_sizes "20,20"
python run_SAE1.py --dataset_name simulation1a --model_name sae --sample False --input_size 100 --dropout 0.5 --epochs 10 --learning_rate 0.001 --sparsity_penalty 1e-4 --hidden_sizes "30,30"
python run_SAE1.py --dataset_name simulation1a --model_name sae --sample False --input_size 100 --dropout 0.5 --epochs 10 --learning_rate 0.001 --sparsity_penalty 1e-4 --hidden_sizes "50,50"
python run_SAE1.py --dataset_name simulation1a --model_name sae --sample False --input_size 100 --dropout 0.5 --epochs 10 --learning_rate 0.001 --sparsity_penalty 1e-4 --hidden_sizes "100,100"