import argparse

VERSION = "Diablo AI Tool v1.5"


def get_parser():
    class IndentedHelpFormatter(argparse.RawTextHelpFormatter):
        def __init__(self, *args, **kwargs):
            # Width controls line wrapping; max_help_position controls indent
            kwargs["max_help_position"] = 8
            super().__init__(*args, **kwargs)

    # Define incompatible options
    incompatible_options = {
        "--attach": [
            "--game-ticks-per-step",
            "--step-mode",
            "--no-monsters",
            "--harmless-barrels",
            "--seed",
            "--fixed-seed",
        ]
    }

    parser = argparse.ArgumentParser(
        prog="diablo-ai.py",
        description=(
            VERSION + "\n\n"
            "Tool which trains AI for playing Diablo and helps to evalulate and play as human.\n\n"
        ),
        epilog="For more details, see https://github.com/rouming/DevilutionX-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=VERSION)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common reusable options
    common_parser = argparse.ArgumentParser(add_help=False)
    # See also `incompatible_options`
    common_parser.add_argument(
        "--attach",
        metavar="MEM_PATH_OR_PID",
        help=(
            "Attach to existing Diablo instance by path, pid or an index of an instance from the `diablo-ai.py list` output. For example:\n"
            "  Attach by PID:\n"
            "    diablo-ai.py play --attach 112342\n"
            "\n"
            "  Attach by path:\n"
            "    diablo-ai.py play --attach /tmp/diablo-tj3bxyvy/shared.mem\n"
            "\n"
            "  Attach by index:\n"
            "    diablo-ai.py play --attach 0"
        ),
    )
    common_parser.add_argument(
        "--view-radius",
        type=int,
        default=10,
        help="Number of environment cells surrounding the AI agent. Set to 0 if want the whole dungeon (default: 10)",
    )
    common_parser.add_argument(
        "--game-ticks-per-step",
        type=int,
        default=10,
        help="Number of game ticks per a single step (default: 10)",
    )
    common_parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run the game loop in real-time, as opposed to step mode (default: false)",
    )
    common_parser.add_argument(
        "--gui", action="store_true", help="Start Diablo in GUI mode only"
    )
    # See also `incompatible_options`
    common_parser.add_argument(
        "--no-monsters", action="store_true", help="Disable all monsters on the level"
    )
    common_parser.add_argument(
        "--device", default="cuda", help="what device to train on"
    )
    common_parser.add_argument(
        "--disable-loot",
        action="store_true",
        help="Disable all loot (items only). Useful to avoid full inventories",
    )
    # See also `incompatible_options`
    common_parser.add_argument(
        "--harmless-barrels",
        action="store_true",
        help="Disable explosive barrels, urns, or pods",
    )
    # See also `incompatible_options`
    common_parser.add_argument(
        "--seed", type=int, default=0, help="Initial seed (default: 0)."
    )
    common_parser.add_argument(
        "--max_episode_length",
        type=int,
        default=1000,
        help="Initial seed (default: 0).",
    )
    # See also `incompatible_options`
    common_parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="Every new game starts with the same seed, so the game world (dungeon) is identical each time.",
    )

    train_rl_parser = subparsers.add_parser(
        "train",
        parents=[common_parser],
        help="Train the RL model by creating new workers and Diablo instances (devilutionX processes), or attach to a single existing instance by providing the `--attach` option (convenient for debug purposes).",
        formatter_class=IndentedHelpFormatter,
    )
    train_rl_parser.add_argument(
        "--algo",
        choices=["dqn"],
        default="dqn",
    )
    train_rl_parser.add_argument(
        "--env",
        help="name of the environment to train on",
        default="Diablo-FindNextLevelWithEnemies-v0",
    )
    train_rl_parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Number of updates between two saves; 0 means no saving (default: 10)",
    )
    train_rl_parser.add_argument(
        "--n_envs",
        type=int,
        default=3,
        help="Number of environment runners or processes (default: 1)",
    )
    train_rl_parser.add_argument(
        "--max_env_steps",
        type=int,
        default=1_000_000,
        help="Number of environment steps for training (default: 10M)",
    )
    train_rl_parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for PPO (default: 256)"
    )
    train_rl_parser.add_argument(
        "--discount", type=float, default=0.99, help="Discount factor (default: 0.99)"
    )
    train_rl_parser.add_argument(
        "--mlp_hidden_dim", type=int, default=128, help="Hidden dimensions of the MLP"
    )
    train_rl_parser.add_argument(
        "--lstm_hidden_dim", type=int, default=128, help="Hidden dimensions of the LSTM"
    )
    train_rl_parser.add_argument(
        "--lr", type=float, default=0.0003, help="Learning rate (default: 0.001)"
    )
    train_rl_parser.add_argument(
        "--epsilon_decay_steps",
        type=int,
        default=10_000,
        help="Learning rate (default: 0.001)",
    )
    train_rl_parser.add_argument(
        "--buffer_size",
        type=int,
        default=int(1e5),
        help="Learning rate (default: 0.001)",
    )
    train_rl_parser.add_argument(
        "--start_training",
        type=int,
        default=5_000,
        help="After how many samples start training the networks",
    )
    train_rl_parser.add_argument(
        "--recurrence_length",
        type=int,
        default=20,
        help="Number of time-steps gradient is backpropagated; If > 1, a LSTM is added to the model to have memory (default: 1)",
    )
    train_rl_parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="tau for target netoworks",
    )
    train_rl_parser.add_argument(
        "--target_net_update_freq",
        type=int,
        default=1,
        help="how often to update the target net towards the live network",
    )
    train_rl_parser.add_argument(
        "--burn_in_phase",
        type=int,
        default=0,
        help="Number of time-steps gradient is backpropagated; If > 1, a LSTM is added to the model to have memory (default: 1)",
    )
    train_rl_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes used to evaluate the agent (default: 10)",
    )
    train_rl_parser.add_argument(
        "--lstm_cell",
        type=int,
        default=0,
        help="Number of episodes used to evaluate the agent (default: 10)",
    )
    train_rl_parser.add_argument(
        "--enemies_deal_no_damage",
        action="store_true",
        help="If active, enemies deal no damage",
    )
    train_rl_parser.add_argument(
        "--log_interval",
        type=int,
        default=1_000,
    )
    train_rl_parser.add_argument(
        "--eval_interval",
        type=int,
        default=500_000,
    )

    eval_rl_parser = subparsers.add_parser(
        "eval",
        parents=[common_parser],
        help="Eval the trained RL agent",
        formatter_class=IndentedHelpFormatter,
    )

    eval_rl_parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path of the trained model",
    )
    eval_rl_parser.add_argument(
        "--env",
        help="name of the environment to eval on",
        default="Diablo-FindNextLevelWithEnemies-v0",
    )

    collect_experience = subparsers.add_parser(
        "collect_experience",
        parents=[common_parser],
        help="Collect bot experience from the env.",
        formatter_class=IndentedHelpFormatter,
    )
    collect_experience.add_argument(
        "--bot", help="name of the bot to use", default="ExploreLevel_Bot"
    )
    collect_experience.add_argument(
        "--env",
        help="name of the environment to train on",
        default="Diablo-FindNextLevelWithEnemies-v0",
    )
    collect_experience.add_argument(
        "--enemies-deal-no-damage",
        action="store_true",
        help="If active, enemies deal no damage",
    )

    play_parser = subparsers.add_parser(
        "play",
        parents=[common_parser],
        help="Train the RL model by creating new workers and Diablo instances (devilutionX processes), or attach to a single existing instance by providing the `--attach` option (convenient for debug purposes).",
        formatter_class=IndentedHelpFormatter,
    )

    return incompatible_options, parser
