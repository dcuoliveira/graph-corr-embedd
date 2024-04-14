
python run_SAE.py --model_name sae --sample False --input_size 100 --dropout 0.5 --epochs 10 --learning_rate 0.001 --sparsity_penalty 1e-4 --hidden_sizes [30,15,30]
python run_SAE.py --model_name sae --sample False --input_size 100 --dropout 0.5 --epochs 100 --learning_rate 0.001 --sparsity_penalty 1e-4 --hidden_sizes [30,15,30]
python run_SAE.py --model_name sae --sample False --input_size 100 --dropout 0.5 --epochs 1000 --learning_rate 0.001 --sparsity_penalty 1e-4 --hidden_sizes [30,15,30]