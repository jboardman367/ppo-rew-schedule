import numpy as np
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
            0 if first_car.on_ground else 1,
            np.linalg.norm(first_car.physics.linear_velocity)
        ]
    
    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        num_touches = 0
        num_goals = 0
        touch_height = 0
        air_time = 0
        velocity = 0
        for metric_array in collected_metrics:
            num_touches += metric_array[0]
            num_goals += metric_array[1]
            touch_height += metric_array[2]
            air_time += metric_array[3]
            velocity += metric_array[4]
        report = {
            "stat/Touch Period (s)": len(collected_metrics) / num_touches / 15,
            "stat/Goal Period (s)": len(collected_metrics) / num_goals / 15,
            "stat/Touch Height (uu)": touch_height / num_touches,
            "stat/Time in air (%)": air_time / len(collected_metrics),
            "stat/Average Velocity (uu/s)": velocity / len(collected_metrics),
        }
        wandb_run.log(report)


def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
    from rlgym.rocket_league.reward_functions import GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine, RLViserRenderer
    from rlgym.rocket_league.state_mutators import MutatorSequence
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    from reward_funcs.zip_reward import ZipReward
    from reward_funcs.vbg import VBG
    from reward_funcs.vpb import VPB
    from reward_funcs.touch_height import TouchHeight
    from reward_funcs.power_shot import PowerShot
    from state_mutators.replay_loader import ReplayLoader
    from state_mutators.variable_team_size import VariableTeamSizeMutator
    from purse_action import PurseAction
    from purse_obs import PurseObs
    from logspace_timeout import LogspaceTimeoutCondition
    import numpy as np

    tick_skip = 8

    action_parser = RepeatAction(PurseAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = LogspaceTimeoutCondition(30, 90)

    reward_fn = ZipReward(
        GoalReward(),
        TouchReward(),
        VPB(),
        VBG(required_touch=True),
        TouchHeight(),
        PowerShot(),
    )

    obs_builder = PurseObs(zero_padding=3,
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    state_mutator = MutatorSequence(VariableTeamSizeMutator(1/6, 2/6, 3/6),
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

    touch_rate = 1 / (4 * 15)
    air_touch_rate = 1 / (10 * 15)
    power_shot_rate = 1 / (15 * 15)

    learner = Learner(build_rlgym_v2_env,
                      metrics_logger=AdditionalMetrics(),
                      checkpoints_save_folder='purse_checkpoints\\v3.0',
                      n_checkpoints_to_keep=50,
                      add_unix_timestamp=False,
                      checkpoint_load_folder='purse_checkpoints\\v3.0\\1132358354',
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      ppo_batch_size=600_000,
                      ts_per_iteration=600_000,
                      exp_buffer_size=1_200_000,
                      ppo_minibatch_size=200_000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      policy_lr=1e-4,
                      critic_lr=1e-4,
                      policy_layer_sizes= (512, 512, 512, 512),
                      critic_layer_sizes= (512, 512, 512, 512),
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=10_000_000,
                      timestep_limit=1_000_000_000_000_000,
                      log_to_wandb=True,
                      load_wandb=True,
                      wandb_group_name='Purse',
                      wandb_run_name='v3.0',
                      reward_scale_config=(
                          (5 * goal_rate, 50 * goal_rate, 200 * goal_rate, 'gc'),
                          (0.05 * touch_rate, 0.4 * touch_rate, 2 * touch_rate, 'touch'),
                          (0.01, 0.1, 0.5, 'vpb'),
                          (0.1 * touch_rate, 1 * touch_rate, 5 * touch_rate, 'touch_vbg'),
                          (0.3 * air_touch_rate, 1.5 * air_touch_rate, 8 * air_touch_rate, 'touch_height'),
                          (0.1 * power_shot_rate, 0.8 * power_shot_rate, 4 * power_shot_rate, 'power_shot')
                      ))
    learner.learn()