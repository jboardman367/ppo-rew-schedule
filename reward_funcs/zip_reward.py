from typing import Any, Dict, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

class ZipReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, *rew_funcs:RewardFunction):
        self.rew_funcs = rew_funcs

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for reward_func in self.rew_funcs:
            reward_func.reset(initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        combined_rewards = {agent: [] for agent in agents}
        for reward_func in self.rew_funcs:
            rewards = reward_func.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent, reward in rewards.items():
                combined_rewards[agent].append(reward)

        return combined_rewards
