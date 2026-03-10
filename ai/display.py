import devilutionx as dx
import curses
import diablo_state
import numpy as np
import ring
import collections
from PIL import Image

# Map symbols to RGB for GIF rendering
SYMBOL_TO_COLOR = {
    " ": (255, 255, 255),  # white
    ".": (200, 200, 200),  # light gray
    "#": (0, 0, 0),  # black
    "C": (255, 0, 0),  # red
    "S": (0, 255, 0),  # green
    "^": (0, 0, 255),  # blue
    "←": (255, 255, 0),  # yellow
    "⚑": (255, 0, 255),  # magenta
}


def display_env_log(stdscr, envlog):
    if envlog is None:
        return

    height, width = stdscr.getmaxyx()
    logwin_h = height // 2
    logwin_w = width // 4

    h = max(0, logwin_h - 2)
    w = max(0, logwin_w - 2)

    # Sane limitation
    if h < 10 or w < 20:
        return

    logwin = stdscr.subwin(logwin_h, logwin_w, 4, 1)

    if envlog.queue is None:
        queue = collections.deque(maxlen=h)
    elif envlog.queue.maxlen != logwin_h:
        queue = collections.deque(maxlen=h)
        for line in envlog.queue:
            queue.append(line)
    else:
        queue = envlog.queue

    while True:
        line = envlog.fd.readline()
        if not line:
            break
        queue.append(line)

    logwin.clear()
    logwin.border()
    msg = " Environment log "
    _addstr(logwin, 0, w // 2 - len(msg) // 2, msg)
    for i, line in enumerate(queue):
        line = truncate_line(line.strip(), w)
        _addstr(logwin, i + 1, 1, line)
    logwin.refresh()

    envlog.queue = queue


def get_events_as_string(game, events):
    advance_progress = False
    while (event := game.retrieve_event()) is not None:
        keys = event.en_type
        k = None

        if keys == 0:
            # Stand, "◦" - white bullet
            k = "\u25e6"
        elif keys == (
            ring.RingEntryType.RING_ENTRY_KEY_UP
            | ring.RingEntryType.RING_ENTRY_KEY_RIGHT
        ):
            # N, "↑" - upwards arrow
            k = "\u2191"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
            # NE, "↗" - north east arrow
            k = "\u2197"
        elif keys == (
            ring.RingEntryType.RING_ENTRY_KEY_DOWN
            | ring.RingEntryType.RING_ENTRY_KEY_RIGHT
        ):
            # E, "→" - rightwards arrow
            k = "\u2192"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN):
            # SE, "↘" - south east arrow
            k = "\u2198"
        elif keys == (
            ring.RingEntryType.RING_ENTRY_KEY_DOWN
            | ring.RingEntryType.RING_ENTRY_KEY_LEFT
        ):
            # S, "↓" - downwards arrow
            k = "\u2193"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_LEFT):
            # SW, "↙" - southwest arrow
            k = "\u2199"
        elif keys == (
            ring.RingEntryType.RING_ENTRY_KEY_UP
            | ring.RingEntryType.RING_ENTRY_KEY_LEFT
        ):
            # W, "←" - leftwards arrow
            k = "\u2190"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_UP):
            # NW, "↖" - north west arrow
            k = "\u2196"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_X):
            k = "X"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_Y):
            k = "Y"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_A):
            k = "A"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_B):
            k = "B"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_NEW):
            k = "N"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_SAVE):
            k = "S"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_LOAD):
            k = "L"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_PAUSE):
            k = "P"

        if k is not None:
            events.queue.append(k)
            advance_progress = True

    if advance_progress:
        events.progress_cnt += 1

    cnt = 0
    s = ""
    for k in events.queue:
        s += " " + k
        cnt += 1

    events_str = " ." * (events.queue.maxlen - cnt) + s
    events_progress = chr(events.progress[events.progress_cnt % len(events.progress)])

    return events_str, events_progress


def _addstr(o, y, x, text):
    try:
        o.addstr(y, x, text)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise


def truncate_line(line, N, extra="..."):
    if N <= len(extra):
        return ""
    return line[: N - len(extra)] + extra if len(line) > N else line


def get_radius(d, dunwin):
    height, width = dunwin.getmaxyx()
    dundim = diablo_state.dungeon_dim(d)

    # Compensate for `R * 2 + _1_` (see `EnvRect`).
    # We use `- 2` because of quarter
    width = min(width, dundim[0]) - 2
    height = min(height, dundim[1]) - 1

    # Reduce the horizontal radius by half to make the dungeon
    # visually appear as an accurate square when displayed in a
    # terminal
    return min(width // 4, height // 2)


# See the comment for the _addstr
def _addch(o, y, x, ch):
    try:
        o.addch(y, x, ch)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise


def display_matrix(dunwin, m):
    cols, rows = m.shape

    # The horizontal radius is reduced by half (see `get_radius()`),
    # so in order to stretch the dungeon number of columns is
    # multiplied by two
    cols *= 2

    # Get the screen size
    height, width = dunwin.getmaxyx()

    x_off = width // 2 - cols // 2
    y_off = height // 2 - rows // 2

    assert x_off >= 0
    assert y_off >= 0

    for row in range(rows):
        for col in range(0, cols, 2):
            _addch(dunwin, row + y_off, col + x_off, m[col // 2, row])
            # "Stretch" the width by adding a space. With this simple
            # trick the dungeon should visually appear as an accurate
            # square in a terminal
            _addch(dunwin, row + y_off, col + x_off + 1, " ")


def display_dungeon(d, stdscr, view_radius, goal_pos):
    height, width = stdscr.getmaxyx()
    dunwin = stdscr.subwin(height - (4 + 1), width, 4, 0)
    radius = get_radius(d, dunwin)
    if view_radius:
        radius = min(radius, view_radius)
    surroundings = diablo_state.get_surroundings(d, radius, goal_pos)

    display_matrix(dunwin, surroundings)


def get_diablo_state(game, stdscr, view_radius):
    d = game.safe_state
    pos = diablo_state.player_position(d)
    return get_dungeon_as_array(d, stdscr, view_radius, game.goal_pos)


def get_dungeon_as_array(game, view_radius):
    d = game.safe_state
    surroundings = diablo_state.get_surroundings(d, view_radius, game.goal_pos)
    return surroundings


def display_diablo_state(game, stdscr, events, envlog, view_radius):
    d = game.safe_state
    pos = diablo_state.player_position(d)

    # Get the screen size
    height, width = stdscr.getmaxyx()

    msg = "Diablo ticks: %4d; Kills: %003d; HP: %d; Pos: %d:%d; State: %-18s" % (
        game.ticks(d),
        np.sum(d.MonsterKillCounts),
        d.player._pHitPoints,
        pos[0],
        pos[1],
        dx.PLR_MODE(d.player._pmode).name,
    )
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 0, width // 2 - len(msg) // 2, msg)

    msg = "Press 'q' to quit"
    _addstr(stdscr, height - 1, width // 2 - len(msg) // 2, msg)

    msg = "Animation: ticksPerFrame %2d; tickCntOfFrame %2d; frames %2d; frame %2d" % (
        d.player.AnimInfo.ticksPerFrame,
        d.player.AnimInfo.tickCounterOfCurrentFrame,
        d.player.AnimInfo.numberOfFrames,
        d.player.AnimInfo.currentFrame,
    )
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 1, width // 2 - len(msg) // 2, msg)

    obj_cnt = diablo_state.count_active_objects(d)
    items_cnt = diablo_state.count_active_items(d)
    total_hp = diablo_state.count_active_monsters_total_hp(d)
    events_str, events_progress = get_events_as_string(game, events)

    msg = "Total: mons HP %d, items %d, objs %d, lvl %d %c %s" % (
        total_hp,
        items_cnt,
        obj_cnt,
        d.player.plrlevel,
        events_progress,
        events_str,
    )
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 2, width // 2 - len(msg) // 2, msg)

    display_dungeon(d, stdscr, view_radius, game.goal_pos)
    display_env_log(stdscr, envlog)

    if diablo_state.is_game_paused(d):
        msgs = [
            "            ",
            " ┌────────┐ ",
            " │ Paused │ ",
            " └────────┘ ",
            "            ",
        ]
        h = height // 2
        for i, msg in enumerate(msgs):
            _addstr(stdscr, h + i, width // 2 - len(msg) // 2, msg)


def dungeon_array_to_image(dungeon_array, scale=20):
    """Convert a dungeon array to a PIL image, correcting rotation."""
    # Transpose the array so (row, col) -> (x, y)
    dungeon_array = dungeon_array.T  # swap axes

    h, w = dungeon_array.shape
    img_array = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            img_array[i, j] = SYMBOL_TO_COLOR.get(dungeon_array[i, j], (255, 255, 255))

    img = Image.fromarray(img_array)
    img = img.resize((w * scale, h * scale), resample=Image.NEAREST)
    return img
