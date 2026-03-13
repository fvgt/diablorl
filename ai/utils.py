import signal
import sys
import resource
import curses
import collections
import os
import time


class EnvLog:
    fd = None
    queue = None

    def __init__(self, fd):
        self.fd = fd


class EventsQueue:
    queue = None
    # Use Braille patterns for representing progress,
    # see here: https://www.unicode.org/charts/nameslist/c_2800.html
    progress = [0x2826, 0x2816, 0x2832, 0x2834]
    progress_cnt = 0

    def __init__(self):
        self.queue = collections.deque(maxlen=10)


def set_sighandlers():
    # Silently terminate on Ctrl-C
    def do_exit(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, do_exit)


def set_rlimits():
    # Get current limits
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set new limits
    new_soft = min(65535, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


def delayed_import(binary_path):
    import devilutionx_generator

    devilutionx_generator.generate(binary_path)

    global dx
    global diablo_env
    global diablo_state
    global diablo_bot
    global ring

    # First goes generated devilutionx
    import devilutionx as dx

    # Then others in any order
    import diablo_env
    import diablo_state
    import diablo_bot
    import ring


def open_envlog(game):
    path = os.path.join(game.state_path, "env.log")
    fd = None
    try:
        fd = open(path, "r")
        return EnvLog(fd)
    except:
        pass
    return None


def set_seed(seed):
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_dicts_to_array(info):
    """
    Flatten any nested dict/tuple/list structure into a single dict of lists.
    Each key maps to a list of all corresponding values.
    """
    merged = {}

    def flatten(x):
        if isinstance(x, dict):
            for k, v in x.items():
                merged.setdefault(k, []).append(v)
        elif isinstance(x, (list, tuple)):
            for item in x:
                flatten(item)
        # ignore anything else

    flatten(info)
    return merged
