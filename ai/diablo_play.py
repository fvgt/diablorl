import diablo_state
from display import display_diablo_state
import curses
from utils import EventsQueue, open_envlog
import ring
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
