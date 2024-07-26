import pickle


def save_pickle(obj, p):
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(p):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj
