from typing import Any, Dict, Union
import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.common_values import BLUE_TEAM

BALL_POS = slice(0, 3)
BALL_LIN_VEL = slice(3, 6)
BALL_ANG_VEL = slice(6, 9)

CARS = slice(9, None)
CAR_POS = slice(0, 3)
CAR_ROT = slice(3, 6)
CAR_LIN_VEL = slice(6, 9)
CAR_ANG_VEL = slice(9, 12)
CAR_BOOST = 12

CAR_STRIDE = 12

class ReplayLoader(StateMutator[GameState]):
    def __init__(
        self,
        ones: Union[np.ndarray, str, None],
        twos: Union[np.ndarray, str, None],
        threes: Union[np.ndarray, str, None]
    ):
        self.ones = ones if ones and isinstance(ones, np.ndarray) else np.load(ones)
        self.twos = twos if twos and isinstance(twos, np.ndarray) else np.load(twos)
        self.threes = threes if threes and isinstance(threes, np.ndarray) else np.load(threes)

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if len(state.cars) == 2:
            replay = self.ones[np.random.randint(0, len(self.ones))]
        elif len(state.cars) == 4:
            replay = self.twos[np.random.randint(0, len(self.twos))]
        elif len(state.cars) == 6:
            replay = self.threes[np.random.randint(0, len(self.threes))]
        else:
            raise ValueError('State must be 1v1, 2v2, or 3v3')

        # Place ball
        state.ball.position = replay[BALL_POS]
        state.ball.linear_velocity = replay[BALL_LIN_VEL]
        state.ball.angular_velocity = replay[BALL_ANG_VEL]

        team_size = len(state.cars) // 2
        blue_cars = 0
        orange_cars = 0
        car_arrays = np.split(replay[CARS], len(state.cars))
        # Place cars
        for car in state.cars.values():
            if car.team_num == BLUE_TEAM:
                self._load_car(car, car_arrays[blue_cars])
                blue_cars += 1
            else:
                self._load_car(car, car_arrays[team_size + orange_cars])
                orange_cars += 1


    def _load_car(self, car: Car, data: np.ndarray):
        car.physics.position = data[CAR_POS]
        car.physics.euler_angles = data[CAR_ROT]
        car.physics.linear_velocity = data[CAR_LIN_VEL]
        car.physics.angular_velocity = data[CAR_ANG_VEL]
        car.boost_amount = data[CAR_BOOST]
