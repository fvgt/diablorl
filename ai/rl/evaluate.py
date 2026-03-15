import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from rl.env.parallel_env import ParallelEnv
import devilutionx as dx
import diablo_env

from rl.models.recurrent import RecurrentModel
from rl.agent.dqn import RecurrentDQN
import torch
from rl.utils.evaluate import evaluate
from rl.utils.load_utils import load_from_pkl
from rl.agent.agent_wrapper import InputNormalizationWrapper
from diablo_env import ActionEnum

import diablo_state
from display import display_diablo_state
import curses
from utils import EventsQueue, open_envlog
import ring
import torch
import time
from rl.utils.action_mask import build_action_mask_fn
import numpy as np

RUNNING = True
LAST_KEY = 0


def get_env_status(d):
    monsters_cnt = diablo_state.count_active_monsters(d)
    hp = d.player._pHitPoints
    mode = d.player._pmode
    player_pos = diablo_state.player_position(d)
    return np.array([monsters_cnt, hp, mode, *player_pos], dtype=np.uint32)


def getting_agent_action(agent, memory, obs, game_features, get_action_mask):
    global LAST_KEY
    global RUNNING

    action_mask = get_action_mask(
        obs=obs,
    )
    action, memory = agent.sample(
        obs=obs, game_features=game_features, memory=memory, masks=action_mask
    )

    # from numpy to int
    action = action.item()

    action_to_key = {
        ActionEnum.Walk_N.value: ring.RingEntryType.RING_ENTRY_KEY_UP,
        ActionEnum.Walk_NE.value: ring.RingEntryType.RING_ENTRY_KEY_UP,  # or custom diagonal key
        ActionEnum.Walk_E.value: ring.RingEntryType.RING_ENTRY_KEY_RIGHT,
        ActionEnum.Walk_SE.value: ring.RingEntryType.RING_ENTRY_KEY_DOWN,  # diagonal mapping
        ActionEnum.Walk_S.value: ring.RingEntryType.RING_ENTRY_KEY_DOWN,
        ActionEnum.Walk_SW.value: ring.RingEntryType.RING_ENTRY_KEY_DOWN,  # diagonal mapping
        ActionEnum.Walk_W.value: ring.RingEntryType.RING_ENTRY_KEY_LEFT,
        ActionEnum.Walk_NW.value: ring.RingEntryType.RING_ENTRY_KEY_UP,  # diagonal mapping
        ActionEnum.Stand.value: None,  # no key
        ActionEnum.PrimaryAction.value: ring.RingEntryType.RING_ENTRY_KEY_A,
        ActionEnum.SecondaryAction.value: ring.RingEntryType.RING_ENTRY_KEY_B,
    }

    # Get key safely
    key = action_to_key.get(action, None)
    LAST_KEY |= key

    return True


def evaluate_trained_agent(stdscr, args, gameconfig):
    global RUNNING
    global LAST_KEY

    lstm_hidden_dim = 128
    agent = load_from_pkl(path=args.model_path)
    memory = torch.zeros(1, lstm_hidden_dim, device=args.device)

    # Run or attach to Diablo
    game = diablo_state.DiabloGame.run_or_attach(gameconfig)

    # create the action mask to filter invalid actions
    n_actions = len(diablo_env.ActionEnum)
    get_action_mask = build_action_mask_fn(gameconfig=gameconfig, n_actions=n_actions)

    # Disable cursor and enable keypad input
    curses.curs_set(0)
    stdscr.nodelay(True)

    events = EventsQueue()
    envlog = None

    # Main loop
    while RUNNING:
        d = game.safe_state
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
        obs = diablo_state.get_environment(d=d, radius=gameconfig["view-radius"])[
            np.newaxis
        ]
        game_features = get_env_status(d=d)[np.newaxis]
        while not getting_agent_action(
            agent=agent,
            memory=memory,
            obs=obs,
            game_features=game_features,
            get_action_mask=get_action_mask,
        ):
            pass

        game.update_ticks()
        time.sleep(0.01)
