import pickle
import pathlib
import torch
import numpy as np


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()

    if isinstance(obj, torch.nn.Module):
        obj = obj.to("cpu")
        return obj

    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)

    # handle custom classes
    if hasattr(obj, "__dict__"):
        if hasattr(obj, "device"):
            setattr(obj, "device", "cpu")
        for k, v in obj.__dict__.items():
            setattr(obj, k, to_cpu(v))
        return obj

    return obj


def save_to_pkl(path, obj, move_to_cpu):
    if move_to_cpu:
        obj = to_cpu(obj)
    file = open(path + ".pkl", "wb")
    # Use protocol>=4 to support saving replay buffers >= 4Gb
    # See https://docs.python.org/3/library/pickle.html
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    if isinstance(path, (str, pathlib.Path)):
        file.close()
