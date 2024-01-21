from typing import Any, Dict, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CEILING_Z, BALL_RADIUS

import numpy as np

class TouchHeight(RewardFunction[AgentID, GameState, float]):
    def __init__(self, require_off_ground=False, min_height=3*BALL_RADIUS):
        self.require_off_ground = require_off_ground
        self.min_height = min_height

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return { agent: self._get_reward(agent, state) for agent in agents }

    def _get_reward(self, agent: AgentID, state: GameState):
        ball = state.ball
        car = state.cars[agent]
        touched = car.ball_touches > 0
        high_enough = ball.position[2] > self.min_height
        invalid_grounded = self.require_off_ground and car.on_ground
        if touched and high_enough and (not invalid_grounded):
            return ball.position[2] / CEILING_Z
        else:
            return 0
    