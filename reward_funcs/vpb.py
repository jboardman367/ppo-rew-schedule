from typing import Any, Dict, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import SUPERSONIC_THRESHOLD

import numpy as np

class VPB(RewardFunction[AgentID, GameState, float]):
    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return { agent: self._get_reward(agent, state) for agent in agents }

    def _get_reward(self, agent: AgentID, state: GameState):
        ball = state.ball
        car = state.cars[agent].physics
    
        scaled_car_vel = car.linear_velocity / SUPERSONIC_THRESHOLD
        car_to_ball = ball.position - car.position
        unit_car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

        return np.dot(unit_car_to_ball, scaled_car_vel)