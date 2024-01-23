from typing import List, Dict, Any
from random import random
from math import log2

from rlgym.api import DoneCondition, AgentID
from rlgym.rocket_league.api import GameState


class LogspaceTimeoutCondition(DoneCondition[AgentID, GameState]):

    def __init__(self, min_sec: float, max_sec: float, tick_rate=1/120):
        """
        :param timeout: Timeout in seconds
        """
        self.min = min_sec
        self.range = max_sec - min_sec
        self.tick_rate = tick_rate
        self.timeout = None
        self.initial_tick = None

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.initial_tick = initial_state.tick_count
        self.timeout = self.min + log2(1 + 9 * random()) * self.range

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        time_elapsed = (state.tick_count - self.initial_tick) * self.tick_rate
        done = time_elapsed >= self.timeout
        return {agent: done for agent in agents}