import pickle

def save_pickle(path: str,
                obj: dict):

    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):

    with open(path, 'rb') as handle:
        target_dict = pickle.load(handle)

    return target_dict