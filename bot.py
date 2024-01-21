from numpy import ndarray
from rlgym_sim.utils.gamestates import GameState

from rlgym.rocket_league.api import GameState
from rlgym_ppo.util import MetricsLogger

class AdditionalMetrics(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        first_car = list(game_state.cars.values())[0]
        return [
            first_car.ball_touches,
            1 if game_state.goal_scored else 0,
            game_state.ball.position[2] * max(0, first_car.ball_touches),
        ]
    
    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        num_touches = 0
        num_goals = 0
        touch_height = 0
        for metric_array in collected_metrics:
            num_touches += metric_array[0]
            num_goals += metric_array[1]
            touch_height += metric_array[2]
        report = {
            "stat/Touch Period (s)": len(collected_metrics) / num_touches / 15,
            "stat/Goal Period (s)": len(collected_metrics) / num_goals / 15,
            "stat/Touch Height (uu)": touch_height / num_touches,
        }
        wandb_run.log(report)


def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine, RLViserRenderer
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    from reward_funcs.zip_reward import ZipReward
    from reward_funcs.vbg import VBG
    from reward_funcs.vpb import VPB
    from reward_funcs.touch_height import TouchHeight
    from state_mutators.replay_loader import ReplayLoader
    from state_mutators.variable_team_size import VariableTeamSizeMutator
    import numpy as np

    tick_skip = 8
    timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout=timeout_seconds)

    reward_fn = ZipReward(
        GoalReward(),
        TouchReward(),
        VPB(),
        VBG(required_touch=True),
        TouchHeight(),
    )

    obs_builder = DefaultObs(zero_padding=3,
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    state_mutator = MutatorSequence(VariableTeamSizeMutator(1/3, 1/3, 1/3),
                                    ReplayLoader('ones-states.npy', 'twos-states.npy', 'threes-states.npy'))
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
    n_proc = 24 # 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    goal_rate = 1 / (35 * 15)

    touch_rate = 1 / (5 * 15)

    learner = Learner(build_rlgym_v2_env,
                      metrics_logger=AdditionalMetrics(),
                      checkpoints_save_folder='purse_checkpoints\\v2.1',
                      add_unix_timestamp=False,
                      checkpoint_load_folder='purse_checkpoints\\v2.1\\204770082',
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      ppo_batch_size=250_000,
                      ts_per_iteration=250_000,
                      exp_buffer_size=750_000,
                      ppo_minibatch_size=250_000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      policy_lr=1e-4,
                      critic_lr=1e-4,
                      policy_layer_sizes= (512, 512, 512),
                      critic_layer_sizes= (512, 512, 512),
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=1_000_000,
                      timestep_limit=1_000_000_000_000,
                      log_to_wandb=True,
                      load_wandb=True,
                      wandb_group_name='Purse',
                      wandb_run_name='v2.1',
                      reward_scale_config=(
                          (1 * goal_rate, 10 * goal_rate, 50 * goal_rate, 'gc'),
                          (0.2 * touch_rate, 1.2 * touch_rate, 6 * touch_rate, 'touch'),
                          (0.001, 0.01, 0.05, 'vpb'),
                          (0.1 * touch_rate, 0.8 * touch_rate, 4 * touch_rate, 'touch_vbg'),
                          (0.2 * touch_rate, 1.2 * touch_rate, 6 * touch_rate, 'touch_height'),
                      ))
    learner.learn()