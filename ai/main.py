import sys
import os
from utils import delayed_import
import configparser
from pathlib import Path
import re


def main():
    config = configparser.ConfigParser()
    config.read("diablo-ai.ini")

    diablo_build_path = Path(config["default"]["diablo-build-path"]).resolve()
    diablo_bin_path = str(diablo_build_path / "devilutionx")

    # important to do this first
    delayed_import(diablo_bin_path)

    from utils import set_rlimits, set_sighandlers
    from parser import get_parser
    import procutils
    import curses
    import utils
    from rl.train import train
    from rl.evaluate import evaluate_trained_agent
    from rl.expert_demonstrations import collect_experience
    from diablo_play import run_tui

    set_sighandlers()
    set_rlimits()

    incompatible_options, parser = get_parser()
    args = parser.parse_args()

    # Check if some options are incompatible
    for opt, incompatibles in incompatible_options.items():
        if opt in sys.argv and (set(incompatibles) & set(sys.argv)):
            parser.error(
                f"{opt} cannot be used together with {' or '.join(incompatibles)}"
            )

    config = configparser.ConfigParser()
    config.read("diablo-ai.ini")

    # Absolute path
    diablo_build_path = Path(config["default"]["diablo-build-path"]).resolve()
    diablo_mshared_filename = config["default"]["diablo-mshared-filename"]

    if not diablo_build_path.is_dir() or len(diablo_mshared_filename) == 0:
        print(
            "Error: initial configuration is invalid. Please check your 'diablo-ai.ini' file and provide valid paths for 'diablo-build-path' and 'diablo-mshared-filename' configuration options."
        )
        sys.exit(1)

    if not (diablo_build_path / "spawn.mpq").exists():
        print(
            f'Error: Shareware file "spawn.mpq" for Diablo content does not exist. Please download and place the file alongside the `devilutionx` binary with the following command:\n\twget -nc https://github.com/diasurgical/devilutionx-assets/releases/download/v2/spawn.mpq -P {diablo_build_path}'
        )
        sys.exit(1)

    diablo_bin_path = str(diablo_build_path / "devilutionx")
    delayed_import(diablo_bin_path)

    # Set seed for all randomness sources
    utils.set_seed(args.seed)

    gameconfig = {
        "mshared-filename": diablo_mshared_filename,
        "diablo-bin-path": diablo_bin_path,
        # Common
        "seed": args.seed,
        "no-monsters": args.no_monsters,
        "harmless-barrels": args.harmless_barrels,
        "no-auto-walk-on-seconday-action": True,  # Changed by old environments
        "view-radius": args.view_radius,
        "game-ticks-per-step": args.game_ticks_per_step,
        "step-mode": not args.real_time,
        "gui": args.gui,
        # AI
        "fixed-seed": args.fixed_seed if hasattr(args, "fixed_seed") else False,
        "log-to-stdout": (
            args.log_to_stdout if hasattr(args, "log_to_stdout") else False
        ),
        "no-actions": args.no_actions if hasattr(args, "no_actions") else False,
        "exploration-door-attraction": (
            args.exploration_door_attraction
            if hasattr(args, "exploration_door_attraction")
            else False
        ),
        "exploration-door-backtrack-penalty": (
            args.exploration_door_backtrack_penalty
            if hasattr(args, "exploration_door_backtrack_penalty")
            else False
        ),
        "enemies-deal-no-damage": (
            args.enemies_deal_no_damage
            if hasattr(args, "enemies_deal_no_damage")
            else False
        ),
        "disable-loot": (args.disable_loot) if hasattr(args, "disable_loot") else False,
        "max_episode_length": (
            (args.max_episode_length) if hasattr(args, "disable_loot") else None
        ),
    }

    if args.attach:
        path_or_pid = args.attach

        if re.match(r"^\d+$", path_or_pid):
            pid_or_index = int(path_or_pid)
            procs = procutils.find_processes_with_mapped_file(
                diablo_bin_path, diablo_mshared_filename
            )
            if pid_or_index < len(procs):
                # Expect index to be a smaller number compared to PID
                index = pid_or_index
                proc = procs[index]
                gameconfig["attach-path"] = proc["mshared_path"]
                gameconfig["attach-offset"] = proc["offset"]
            else:
                pid = pid_or_index
                mshared_path, offset = procutils.get_mapped_file_and_offset_of_pid(
                    pid, diablo_mshared_filename
                )
                if mshared_path:
                    gameconfig["attach-path"] = mshared_path
                    gameconfig["attach-offset"] = offset

        elif os.path.exists(path_or_pid):
            mshared_path = path_or_pid
            procs = procutils.find_processes_with_mapped_file(
                diablo_bin_path, mshared_path
            )
            if len(procs) == 1:
                gameconfig["attach-path"] = mshared_path
                gameconfig["attach-offset"] = procs[0]["offset"]

        if "attach-path" not in gameconfig or "attach-offset" not in gameconfig:
            print(
                "Error: --attach=%s is not a valid path, PID or index of a Diablo instance"
                % path_or_pid
            )
            sys.exit(1)

    if args.command == "play":
        curses.wrapper(lambda stdscr: run_tui(stdscr, args, gameconfig))
    elif args.command == "collect_experience":
        collect_experience(args, gameconfig)
    elif args.command == "train":
        train(args=args, gameconfig=gameconfig)
    elif args.command == "eval":
        curses.wrapper(lambda stdscr: evaluate_trained_agent(stdscr, args, gameconfig))


if __name__ == "__main__":
    main()
