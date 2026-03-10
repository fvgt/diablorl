import signal
import sys
import resource
import curses
import collections
from display import display_diablo_state
import os
import time

RUNNING = True
LAST_KEY = 0


def handle_keyboard(stdscr):
    global LAST_KEY
    global RUNNING

    k = stdscr.getch()
    if k == -1:
        return False

    key = 0

    if k == 259:
        key = ring.RingEntryType.RING_ENTRY_KEY_UP
    elif k == 258:
        key = ring.RingEntryType.RING_ENTRY_KEY_DOWN
    elif k == 260:
        key = ring.RingEntryType.RING_ENTRY_KEY_LEFT
    elif k == 261:
        key = ring.RingEntryType.RING_ENTRY_KEY_RIGHT
    elif k == ord("a"):
        key = ring.RingEntryType.RING_ENTRY_KEY_A
    elif k == ord("b"):
        key = ring.RingEntryType.RING_ENTRY_KEY_B
    elif k == ord("x"):
        key = ring.RingEntryType.RING_ENTRY_KEY_X
    elif k == ord("y"):
        key = ring.RingEntryType.RING_ENTRY_KEY_Y
    elif k == ord("n"):
        key = ring.RingEntryType.RING_ENTRY_KEY_NEW
    elif k == ord("l"):
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD
    elif k == ord("s"):
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE
    elif k == ord("p"):
        key = ring.RingEntryType.RING_ENTRY_KEY_PAUSE
    elif k == ord("q"):
        RUNNING = False  # Stop the main loop

    LAST_KEY |= key

    return True


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


def run_tui(stdscr, args, gameconfig):
    global RUNNING
    global LAST_KEY

    # Run or attach to Diablo
    game = diablo_state.DiabloGame.run_or_attach(gameconfig)

    # Disable cursor and enable keypad input
    curses.curs_set(0)
    stdscr.nodelay(True)

    events = EventsQueue()
    envlog = None

    # Main loop
    while RUNNING:
        stdscr.clear()

        envlog = open_envlog(game)
        view_radius = args.view_radius

        display_diablo_state(game, stdscr, events, envlog, view_radius)

        if LAST_KEY:
            key = LAST_KEY
            key |= ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            game.submit_key(key)
            LAST_KEY = 0

        # Refresh the screen to show the content
        stdscr.refresh()

        # Handle keys
        while handle_keyboard(stdscr):
            pass

        game.update_ticks()
        time.sleep(0.01)


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
