from numpy import ndarray
from rlgym_sim.utils.gamestates import GameState
from reward_funcs.zip_reward import ZipReward
from reward_funcs.vbg import VBG
from reward_funcs.vpb import VPB

from rlgym.rocket_league.api import GameState
from rlgym_ppo.util import MetricsLogger

class AdditionalMetrics(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        first_car = list(game_state.cars.values())[0]
        return [
            first_car.ball_touches,
            1 if game_state.goal_scored else 0
        ]
    
    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        num_touches = 0
        num_goals = 0
        for metric_array in collected_metrics:
            num_touches += metric_array[0]
            num_goals += metric_array[1]
        report = {
            "stat/Touch Period (s)": len(collected_metrics) / num_touches / 15,
            "stat/Goal Period (s)": len(collected_metrics) / num_goals / 15,
        }
        wandb_run.log(report)


def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine, RLViserRenderer
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    import numpy as np

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    tick_skip = 8
    timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout=timeout_seconds)

    reward_fn = ZipReward(GoalReward(), TouchReward(), VPB(), VBG(required_touch=True))

    obs_builder = DefaultObs(zero_padding=None,
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    state_mutator = MutatorSequence(FixedTeamSizeMutator(blue_size=blue_team_size,
                                                                  orange_size=orange_team_size),
                                    KickoffMutator())
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer())

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    goal_rate = 1 / (35 * 15)

    touch_rate = 1 / (5 * 15)

    learner = Learner(build_rlgym_v2_env,
                      metrics_logger=AdditionalMetrics(),
                      checkpoints_save_folder='data\\v1',
                      checkpoint_load_folder='data\\v1-1705154107679511500\\53853366',
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=True,
                      load_wandb=True,
                      wandb_run_name='v1',
                      reward_scale_config=(
                          (1 * goal_rate, 10 * goal_rate, 50 * goal_rate, 'gc'),
                          (0.2 * touch_rate, 1.2 * touch_rate, 6 * touch_rate, 'touch'),
                          (0.001, 0.01, 0.05, 'vpb'),
                          (0.1 * touch_rate, 0.8 * touch_rate, 4 * touch_rate, 'touch_vbg')
                      ))
    learner.learn()