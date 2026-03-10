import multiprocessing
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np
import copy
import time
import diablo_state
from rl.wrapper import RenderWrapper, EasyGameAccessWrapper
import utils


def create_envs(env_name, gameconfig, n_envs, seeds):
    # Load environments
    envs = []
    ts = 0
    for i in range(n_envs):
        env_config = copy.deepcopy(gameconfig)
        env_config["seed"] = seeds[i]

        # Some old environments have specific configurations that need
        # to be adjusted before starting a game instance
        EnvClass = get_env_class(env_name)
        EnvClass.tune_config(env_config)

        # Run or attach to Diablo (devilutionX) instance
        game = diablo_state.DiabloGame.run_or_attach(env_config)
        env = make_env(env_name, env_config, game)
        envs.append(env)

        if time.time() - ts >= 3.0 or i == n_envs - 1:
            ts = time.time()
            print(f"{i+1}/{n_envs} environment instances are created")

    return envs


def get_env_class(env_key):
    spec = gym.spec(env_key)
    return spec.entry_point


def make_env(env_key, env_config, game):
    env = gym.make(env_key, env_config=env_config, game=game)
    env = RenderWrapper(env, game_instance=game, view_radius=env_config["view-radius"])
    env = EasyGameAccessWrapper(env, game_instance=game)
    return env


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            result = env.step(data)
            # terminated, truncated = result[2:4]
            # if terminated or truncated:
            # Be careful here - the last observation is returned
            # right after reset, not the actual observation that
            # causes termination. This should not cause any
            # harm for training because the algorithm does not
            # actually use the next observation when done=True.
            # See @ParallelEnv.step()
            # obs, _ = env.reset()
            # result = (obs,) + result[1:]
            conn.send(result)
        elif cmd == "reset":
            obs, info = env.reset(seed=data)
            conn.send((obs, info))
        else:
            raise NotImplementedError


class ParallelEnvPool:
    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."
        self.ctx = multiprocessing.get_context("fork")
        self.envs = envs
        self.locals = []
        for env in self.envs[1:]:
            local, remote = self.ctx.Pipe()
            self.locals.append(local)
            p = self.ctx.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, env_name, gameconfig, n_envs, seeds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        env_list = create_envs(
            env_name=env_name, gameconfig=gameconfig, n_envs=n_envs, seeds=seeds
        )
        self.penv_pool = ParallelEnvPool(envs=env_list)

        # For bot environment spaces are missing
        if hasattr(self.penv_pool, "observation_space"):
            env_status_space = self.penv_pool.observation_space["env-status"]
            env_space = self.penv_pool.observation_space["env"]

            self.observation_space = Dict(
                {
                    "env": Box(
                        low=np.repeat(
                            env_space.low[np.newaxis], repeats=n_envs, axis=0
                        ).astype(np.float32),
                        high=np.repeat(
                            env_space.high[np.newaxis], repeats=n_envs, axis=0
                        ).astype(np.float32),
                        shape=(n_envs, *env_space.shape),
                    ),
                    "env_status": Box(
                        low=np.repeat(
                            env_status_space.low[np.newaxis], repeats=n_envs, axis=0
                        ).astype(np.float32),
                        high=np.repeat(
                            env_status_space.high[np.newaxis], repeats=n_envs, axis=0
                        ).astype(np.float32),
                        shape=(n_envs, *env_status_space.shape),  # <- flattened
                    ),
                }
            )

        if hasattr(self.penv_pool.envs[0], "action_space"):
            self.action_space = self.penv_pool.action_space

        self.last_obs = None

        # self.preprocess_obs_fn = preprocess_obs_fn

    def process_env_state(self, env_states):
        obs = np.stack([res["env"] for res in env_states])
        game_features = np.stack([res["env-status"] for res in env_states])
        return obs, game_features

    def reset_where_done(self, observations, infos, terms, truncs, seeds=None):
        for i, (term, trunc) in enumerate(zip(terms, truncs)):
            if term or trunc:
                obs, info = self.reset_single(idx=i, seed=seeds[i])
                observations[i] = obs
                infos["game_features"][i] = info["game_features"]

        return observations, infos

    def reset_single(self, idx, seed=None):
        if idx == 0:
            env_states, info = self.penv_pool.envs[0].reset(seed=seed)
        else:
            self.penv_pool.locals[idx - 1].send(("reset", seed))
            env_states, info = self.penv_pool.locals[idx - 1].recv()
        # pass in list, since process env state expects a list of env_states
        obs, game_features = self.process_env_state([env_states])
        info["game_features"] = game_features
        return obs, info

    def reset(self, seeds=None):
        if seeds is not None:
            assert len(seeds) and len(seeds) <= len(self.penv_pool.envs)
        else:
            seeds = [None] * len(self.penv_pool.envs)

        for local, seed in zip(self.penv_pool.locals, seeds[1:]):
            local.send(("reset", seed))

        results = [self.penv_pool.envs[0].reset(seed=seeds[0])] + [
            local.recv() for local, _ in zip(self.penv_pool.locals, seeds[1:])
        ]
        # results contain empty dict, get rid of it
        env_states = [res[0] for res in results]
        obs, game_features = self.process_env_state(env_states)
        info = dict(game_features=game_features)
        return obs, info

    def step(self, actions):
        assert len(actions) and len(actions) <= len(self.penv_pool.envs)

        for local, action in zip(self.penv_pool.locals, actions[1:]):
            local.send(("step", action))

        result = self.penv_pool.envs[0].step(actions[0])
        env_states, rewards, terminated, truncated, info = zip(
            *[result]
            + [local.recv() for local, _ in zip(self.penv_pool.locals, actions[1:])]
        )
        info = utils.merge_dicts_to_array(info)
        obs, game_features = self.process_env_state(env_states)
        info["game_features"] = game_features
        return (
            obs,
            np.stack(rewards),
            np.stack(terminated),
            np.stack(truncated),
            info,
        )

    def render(self, mode="terminal", idx=0):
        return self.penv_pool.envs[idx].env.render(mode=mode)
