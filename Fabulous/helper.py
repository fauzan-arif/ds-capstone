import pickle

def save_model(model, filename="final_model.h5"):
    return pickle.dump(model, open(filename, "wb"))

def load_model(filename="final_model.h5"):
    return pickle.load(open(filename, "rb"))