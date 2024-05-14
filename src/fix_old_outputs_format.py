import argparse
import os
from tqdm import tqdm

from utils.conn_data import load_pickle, save_pickle

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, help="Model name to fix outputs format", default="sdne0")
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1a")

if __name__ == "__main__":
    args = parser.parse_args()
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}"

    # list files that starts with "model_name"
    files = os.listdir(output_path)
    files = [file for file in files if file.startswith(args.model_name)]
    pbar = tqdm(files, desc=f"Fixing outputs format for model = {args.model_name}", total=len(files))
    for file in pbar:
        result_names = os.listdir(f"{output_path}/{file}")

        if "results.pkl" in result_names:
            results = load_pickle(path=f"{output_path}/{file}/results.pkl")

            if "sdne" in args.model_name:
                old_args = results["args"]
                epochs_predictions = results["train_predictions"]
                test_results = results["test_predictions"]
                epochs_tot_loss = results["train_loss"]
                epochs_global_loss = results["epochs_global_loss"]
                epochs_local_loss = results["epochs_local_loss"]
                epochs_reg_loss = results["epochs_reg_loss"]

                new_args = {
                    "args": old_args
                }

                predictions = {
                    "train_predictions": epochs_predictions,
                    "test_predictions": test_results,
                }

                training_info = {
                    "train_loss": epochs_tot_loss,
                    "epochs_global_loss": epochs_global_loss,
                    "epochs_local_loss": epochs_local_loss,
                    "epochs_reg_loss": epochs_reg_loss,
                }

                save_pickle(path=f"{output_path}/{file}/args.pkl", obj=new_args)
                save_pickle(path=f"{output_path}/{file}/predictions.pkl", obj=predictions)
                save_pickle(path=f"{output_path}/{file}/training_info.pkl", obj=training_info)
