import pickle
import joblib
import os
import gc
import _pickle as cpickle

def save_pickle(path: str,
                obj: dict):
    
    if path.endswith(".joblib"):
        joblib.dump(obj, path)
    else:
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle)

def load_pickle(path: str):

    if path.endswith(".joblib"):
        target_dict = joblib.load(path)
    else:
        with open(path, 'rb') as handle:
            target_dict = pickle.load(handle)

    return target_dict

def save_inputs_piecewise(inputs,
                          embeddings,
                          path: str,
                          sample: bool=False):
    
    total = inputs.shape[0]

    if total != embeddings.shape[0]:
        raise ValueError("Inputs and embeddings must have the same number of samples.")
    
    if sample:
        path = os.path.join(path, "sample_inputs")
    else:
        path = os.path.join(path, "inputs")

    # checkif dir exists
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(total):
        tmp_inputs = {
            "inputs": inputs[i],
            "embeddings": embeddings[i],
        }
        save_pickle(path=os.path.join(path, f"inputs_{i}.pkl"), obj=tmp_inputs)


def load_pickle_fast(path: str):
    gc.disable()
    try:
        if path.endswith(".joblib"):
            target_dict = joblib.load(path)
        else:
            with open(path, 'rb') as handle:
                target_dict = cpickle.load(handle)
    finally:
        gc.enable()

    return target_dict
