from typing import Any, Dict, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM, ORANGE_GOAL_CENTER, BALL_MAX_SPEED

import numpy as np

class VBG(RewardFunction[AgentID, GameState, float]):
    def __init__(self, required_touch=False):
        self.required_touch = required_touch
    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return { agent: self._get_reward(agent, state) for agent in agents }

    def _get_reward(self, agent: AgentID, state: GameState):
        if self.required_touch and state.cars[agent].ball_touches < 1:
            return 0.
        if state.cars[agent].team_num == ORANGE_TEAM:
            ball = state.inverted_ball
        else:
            ball = state.ball
        
        scaled_ball_vel = ball.linear_velocity / BALL_MAX_SPEED
        ball_to_goal = ORANGE_GOAL_CENTER - ball.position
        unit_ball_to_goal = ball_to_goal / np.linalg.norm(ball_to_goal)

        return np.dot(unit_ball_to_goal, scaled_ball_vel)