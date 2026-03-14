import pickle
import pathlib
import os


def save_to_pkl(path, obj):
    file = open(path + ".pkl", "wb")
    # Use protocol>=4 to support saving replay buffers >= 4Gb
    # See https://docs.python.org/3/library/pickle.html
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    if isinstance(path, (str, pathlib.Path)):
        file.close()
