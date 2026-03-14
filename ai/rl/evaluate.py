import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from rl.env.parallel_env import ParallelEnv
import devilutionx as dx

from rl.models.recurrent import RecurrentModel
from rl.agent.dqn import RecurrentDQN
import torch
from rl.utils.evaluate import evaluate
from rl.utils.load_utils import load_from_pkl
from rl.agent.agent_wrapper import InputNormalizationWrapper


def evaluate_trained_agent(args, gameconfig):

    lstm_hidden_dim = 128
    device = "cuda"

    eval_env = ParallelEnv(
        env_name=args.env,
        gameconfig=gameconfig,
        n_envs=1,
        seeds=[args.seed],
    )

    agent = load_from_pkl(path=os.path.join(args.model_path, "test_agent"))
    memory = torch.zeros(1, lstm_hidden_dim, device=device)

    eval_reward, eval_success, gif_path = evaluate(
        env=eval_env, agent=agent, _mem=memory, n_episodes=100000000
    )
