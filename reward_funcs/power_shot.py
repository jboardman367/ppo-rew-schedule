from typing import Any, Dict, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_MAX_SPEED

import numpy as np

class PowerShot(RewardFunction[AgentID, GameState, float]):
    last_state: GameState
    def __init__(self, min_dv=300.):
        self.min_dv = min_dv
    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_state = initial_state
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rews = { agent: self._get_reward(agent, state) for agent in agents }
        self.last_state = state
        return rews

    def _get_reward(self, agent: AgentID, state: GameState):
        if state.cars[agent].ball_touches < 1:
            return 0.
        
        dv = np.linalg.norm(state.ball.linear_velocity - self.last_state.ball.linear_velocity)
        
        return dv / BALL_MAX_SPEED if dv > self.min_dv else 0
