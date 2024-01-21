from typing import Dict, Any

import numpy as np

from rlgym.api import StateMutator
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BLUE_TEAM, OCTANE, ORANGE_TEAM


class VariableTeamSizeMutator(StateMutator[GameState]):

    def __init__(self, ones_prob: float, twos_prob: float, threes_prob: float):
        self.repeat_max_size = 2
        assert abs(1 - ones_prob - twos_prob - threes_prob) < 0.00001  # Probs should add to 1
        self.ones_prob = ones_prob
        self.twos_prob = twos_prob

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        assert len(state.cars) == 0  # This mutator doesn't support other team size mutators

        rand = np.random.rand()
        if rand < self.ones_prob:
            team_size = 1
        elif rand < self.ones_prob + self.twos_prob:
            team_size = 2
        else:
            team_size = 3
        
        # WORKAROUND: Need to make a buffer allocate enough memory by starting with 3s
        if self.repeat_max_size:
            self.repeat_max_size -= 1
            team_size = 3

        for idx in range(team_size):
            car = self._new_car()
            car.team_num = BLUE_TEAM
            state.cars['blue-{}'.format(idx)] = car

        for idx in range(team_size):
            car = self._new_car()
            car.team_num = ORANGE_TEAM
            state.cars['orange-{}'.format(idx)] = car

    def _new_car(self) -> Car:
        car = Car()
        car.hitbox_type = OCTANE

        car.physics = PhysicsObject()

        car.demo_respawn_timer = 0.
        car.on_ground = True
        car.supersonic_time = 0.
        car.boost_amount = 0.
        car.boost_active_time = 0.
        car.handbrake = 0.

        car.has_jumped = False
        car.is_holding_jump = False
        car.is_jumping = False
        car.jump_time = 0.

        car.has_flipped = False
        car.has_double_jumped = False
        car.air_time_since_jump = 0.
        car.flip_time = 0.
        car.flip_torque = np.zeros(3, dtype=np.float32)

        car.is_autoflipping = False
        car.autoflip_timer = 0.
        car.autoflip_direction = 0.
        return car