import pickle
import pathlib
import os


def load_from_pkl(path):
    file = open(path + ".pkl", "rb")
    obj = pickle.load(file)
    if isinstance(path, (str, pathlib.Path)):
        file.close()
    return obj
