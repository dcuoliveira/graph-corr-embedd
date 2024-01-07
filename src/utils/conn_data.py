import pickle
import joblib

def save_pickle(path: str,
                obj: dict):
    
    if path.endswith(".joblib"):
        joblib.dump(obj, path)
    else:
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):

    if path.endswith(".joblib"):
        target_dict = joblib.load(path)
    else:
        with open(path, 'rb') as handle:
            target_dict = pickle.load(handle)

    return target_dict