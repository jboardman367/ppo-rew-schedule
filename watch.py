import os
from time import sleep, perf_counter
from bot import build_rlgym_v2_env
import numpy as np

from rlgym_ppo.ppo import DiscreteFF
import torch

env = build_rlgym_v2_env().rlgym_env
obs = env.reset()
# Make a policy
policy_layer_sizes = (512, 512, 512)

device = 'cuda:0' # 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

policy = DiscreteFF(
    env.observation_space(env.agents[0]) or np.prod(np.size(list(obs.values())[0])),
    env.action_space(env.agents[0]),
    policy_layer_sizes,
    device
)

checkpoint_folder: str = 'purse_checkpoints\\v2.1'
current_checkpoint: int = -1

start = perf_counter()

while(True):
    # Load most recent checkpoint
    max_available = max(os.listdir(checkpoint_folder), key=lambda d: int(d))
    if int(max_available) > current_checkpoint:
        policy.load_state_dict(torch.load(os.path.join(checkpoint_folder, max_available, 'PPO_POLICY.pt')))
        current_checkpoint = int(max_available)
        print(f'Loaded checkpoint: {max_available}')

    obs_dict = env.reset()
    done = False
    while not done:
        # get actions
        obs_keys, obs_values = [], []
        for k, v in obs_dict.items():
            obs_keys.append(k)
            obs_values.append(v)
        actions, _ = policy.get_action(np.stack(obs_values, axis=0))
        actions = actions.numpy().astype(np.float32)
        action_dict = { k: np.array([int(a)]) for k, a in zip(obs_keys, actions) }

        obs_dict, _, terminated_dict, truncated_dict = env.step(action_dict)
        done = list(terminated_dict.values())[0] or list(truncated_dict.values())[0]
        sleep((perf_counter() - start) % (8/120))
        env.render()
