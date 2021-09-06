#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from ..utils import *

try:
    import roboschool
except ImportError:
    pass

def make_robosuite_env(env_id):
    import robosuite as suite
    from robosuite.wrappers.gym_wrapper import GymWrapper
    from robosuite import load_controller_config

    import rlkit.util.hyperparameter as hyp

    base_variant=dict(
        env_variant=dict(
            robot_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],
            obj_keys=['object-state'],
            controller_type='OSC_POSITION_YAW',
            controller_config_update=dict(
                position_limits=[
                    [-0.30, -0.30, 0.75],
                    [0.15, 0.30, 1.15]
                ],
            ),
            env_kwargs=dict(
                ignore_done=False, #True
                horizon=150,

                reward_shaping=True,
                hard_reset=False,
                control_freq=10,
                camera_heights=512,
                camera_widths=512,
                table_offset=[-0.075, 0, 0.8],
                reward_scale=5.0,

                skill_config=dict(
                    skills=['ll'], # only using low level actions

                    aff_penalty_type='add',
                    aff_penalty_fac=15.0, # 5.0

                    task_sketch_ids=None,
                    max_skill_repeats=1,
                    terminate_on_sketch=True,
                    aff_th_for_sketch=0.90,

                    base_config=dict(
                        global_xyz_bounds=[
                            [-0.30, -0.30, 0.80],
                            [0.15, 0.30, 0.95]
                        ],
                        lift_height=0.95,
                        binary_gripper=True,

                        aff_threshold=0.06,
                        aff_type='dense',
                        reach_global=True,
                        aff_tanh_scaling=10.0,
                    ),
                    ll_config=dict(
                        use_ori_params=True,
                    ),
                    reach_config=dict(
                        use_gripper_params=False,
                        local_xyz_scale=[0.0, 0.0, 0.06],
                        use_ori_params=False,
                        max_ac_calls=15,
                    ),
                    grasp_config=dict(
                        global_xyz_bounds=[
                            [-0.30, -0.30, 0.80],
                            [0.15, 0.30, 0.85]
                        ],
                        aff_threshold=0.03,

                        local_xyz_scale=[0.0, 0.0, 0.0],
                        use_ori_params=True,
                        max_ac_calls=20,
                        num_reach_steps=2,
                        num_grasp_steps=3,
                    ),
                    push_config=dict(
                        global_xyz_bounds=[
                            [-0.30, -0.30, 0.80],
                            [0.15, 0.30, 0.85]
                        ],
                        delta_xyz_scale=[0.25, 0.25, 0.05],

                        max_ac_calls=20,
                        use_ori_params=True,

                        aff_threshold=[0.12, 0.12, 0.04],
                    ),
                ),
            ),
        ),
    )

    env_params = dict(
        stack={
            'env_variant.env_type': ['Stack'],
            'env_variant.env_kwargs.full_stacking_bonus': [2.0],
        },
        door={
            'env_variant.env_type': ['Door'],
            'env_variant.env_kwargs.use_latch': [True],
            'env_variant.controller_type': ['OSC_POSITION'],
            'env_variant.controller_config_update.position_limits': [[[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
            'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [[[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
            'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [[[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
            'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [[[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
            'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.15],
            'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
        },
        lift={
            'env_variant.env_type': ['Lift'],
        },
        nut_round={
            'env_variant.env_type': ['NutAssemblyRound'],
            'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
        },
        pnp={
            'env_variant.env_type': [
                'PickPlaceCan',
                # 'PickPlaceBread',
                # 'PickPlaceCanBread',
            ],
            'env_variant.env_kwargs.bin1_pos': [[0.0, -0.25, 0.8]],
            'env_variant.env_kwargs.bin2_pos': [[0.0, 0.28, 0.8]],
            'env_variant.controller_config_update.position_limits': [[[-0.15, -0.50, 0.75], [0.15, 0.50, 1.15]]],
            'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [[[-0.15, -0.50, 0.82], [0.15, 0.50, 1.02]]],
            'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [[[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
            'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [[[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
            'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.0],

            'env_variant.env_kwargs.use_center_of_target_bins_for_lift_pos': [True],
            'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.06]],
            'env_variant.env_kwargs.use_unsolved_objs_for_src_pos': [True],
        },
        wipe={
            'env_variant.env_type': ['Wipe'],
            'env_variant.obj_keys': [['robot0_contact-obs', 'object-state']],
            'algorithm_kwargs.max_path_length': [300],
            'env_variant.controller_type': ['OSC_POSITION'],
            'env_variant.env_kwargs.table_offset': [[0.05, 0, 0.8]],
            'env_variant.controller_config_update.position_limits': [[[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
            'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [[[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
            'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [[[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
            'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [[[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
            'env_variant.env_kwargs.task_config': [
                dict(
                    coverage_factor=0.60,
                    num_markers=150,
                    wipe_contact_reward=0.05,
                    excess_force_penalty_mul=0.01, # 0.1
                    distance_multiplier=0.05,
                    distance_th_multiplier=5.0,
                    task_complete_reward=1.5,
                    early_terminations=False,
                    success_th=1.00,
                ),
            ],
            'env_variant.env_kwargs.skill_config.base_config.aff_threshold': [[0.15, 0.25, 0.03]],
            'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.03]],
            'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [[0.15, 0.25, 0.03]],
            'env_variant.env_kwargs.skill_config.push_config.aff_threshold': [[0.15, 0.25, 0.03]],
        },
        peg_ins={
            'env_variant.env_type': ['PegInHole'],
            'env_variant.controller_config_update.position_limits': [[[-0.30, -0.30, 0.75], [0.15, 0.30, 1.00]]],

            'env_variant.env_kwargs.task_config.limit_init_ori': [True],
            'env_variant.env_kwargs.task_config.cos_weight': [1],
            'env_variant.env_kwargs.task_config.d_weight': [1],
            'env_variant.env_kwargs.task_config.t_weight': [5],
            'env_variant.env_kwargs.task_config.scale_by_cos': [True],
            'env_variant.env_kwargs.task_config.scale_by_d': [True],
            'env_variant.env_kwargs.task_config.cos_tanh_mult': [3.0],
            'env_variant.env_kwargs.task_config.d_tanh_mult': [15.0],
            'env_variant.env_kwargs.task_config.t_tanh_mult': [7.5],

            'env_variant.env_kwargs.task_config.lift_pos_offset': [0.30],
            'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [0.06],
        },
        cleanup={
            'env_variant.env_type': ['Cleanup'],
            'env_variant.env_kwargs.task_config': [
                dict(
                    use_pnp_rew=True,
                    use_push_rew=True,
                    rew_type='sum',
                    push_scale_fac=5.0,
                    use_realistic_obj_sizes=True,
                ),
                # dict(
                #     use_pnp_rew=True,
                #     use_push_rew=True,
                #     num_pnp_objs=2,
                #     num_push_objs=1,
                #     rew_type='sum',
                #     push_scale_fac=5.0,
                #     use_realistic_obj_sizes=True,
                # ),
            ],
        },
        cleanup_twin={
            'env_variant.env_type': ['Cleanup'],
            'env_variant.env_kwargs.task_config': [
                dict(
                    use_pnp_rew=True,
                    use_push_rew=True,
                    rew_type='sum',
                    push_scale_fac=5.0,
                    use_realistic_obj_sizes=True,

                    digital_twin=True,
                ),
            ],

            # digital twin settings
            'env_variant.robot_keys': [['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']],
            'env_variant.env_kwargs.table_offset': [[-0.10, 0, 0.8]],
            'env_variant.env_kwargs.table_full_size': [[0.6, 1.0, 0.05]],
            'env_variant.controller_config_update.position_limits': [[[-0.26, -0.35, 0.75], [0.04, 0.35, 1.15]]],
            'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.95]]],
            'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.85]]],
            'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.85]]],
        },
        push_and_stack_twin={
            'env_variant.env_type': ['PushAndStack'],

            # digital twin settings
            'env_variant.robot_keys': [['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']],
            'env_variant.env_kwargs.table_offset': [[-0.10, 0, 0.8]],
            'env_variant.env_kwargs.table_full_size': [[0.6, 1.0, 0.05]],
            'env_variant.controller_config_update.position_limits': [[[-0.26, -0.35, 0.75], [0.04, 0.35, 1.15]]],
            'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.95]]],
            'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.85]]],
            'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.85]]],

            'env_variant.env_kwargs.stack_only': [False],
        },
        stack_twin={
            'env_variant.env_type': ['PushAndStack'],

            # digital twin settings
            'env_variant.robot_keys': [['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']],
            'env_variant.env_kwargs.table_offset': [[-0.10, 0, 0.8]],
            'env_variant.env_kwargs.table_full_size': [[0.6, 1.0, 0.05]],
            'env_variant.controller_config_update.position_limits': [[[-0.26, -0.35, 0.75], [0.04, 0.35, 1.15]]],
            'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.95]]],
            'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.85]]],
            'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [[[-0.26, -0.35, 0.80], [0.04, 0.35, 0.85]]],

            'env_variant.env_kwargs.stack_only': [True],
        },
    )

    sweeper = hyp.DeterministicHyperparameterSweeper(
        env_params[env_id], default_parameters=base_variant,
    )
    hp_list = sweeper.iterate_hyperparameters()
    assert len(hp_list) == 1
    env_variant = hp_list[0]['env_variant']

    controller_config = load_controller_config(default_controller=env_variant['controller_type'])
    controller_config_update = env_variant.get('controller_config_update', {})
    controller_config.update(controller_config_update)

    robot_type = env_variant.get('robot_type', 'Panda')

    obs_keys = env_variant['robot_keys'] + env_variant['obj_keys']

    env = suite.make(
        env_name=env_variant['env_type'],  # "Lift" try with other tasks like "Stack" and "Door"
        robots=robot_type,  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=True, #False,
        use_camera_obs=False,
        controller_configs=controller_config,

        **env_variant['env_kwargs']
    )

    env = GymWrapper(env, keys=obs_keys)

    return env


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed, rank, episode_life=True):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id in [
            'lift', 'door', 'stack', 'nut_round', 'pnp', 'wipe', 'peg_ins', 'cleanup',
            'cleanup_twin', 'push_and_stack_twin', 'stack_twin',
        ]:
            env = make_robosuite_env(env_id)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class PaddingObsWrapper(gym.Wrapper):
    def __init__(self, env, domain, task):
        gym.Wrapper.__init__(self, env)
        self.domain = domain
        self.task = task
        if self.domain == 'fish' and self.task == 'upright':
            # make upright compatible with swim
            self.observation_space = Box(
                -float('inf'),
                float('inf'),
                (24, ),
                dtype=np.float32,
            )

    def pad_obs(self, obs):
        if self.domain == 'fish' and self.task == 'upright':
            new_obs = np.zeros((24, ))
            new_obs[:8] = obs[:8]
            new_obs[11:] = obs[8:]
            return new_obs
        else:
            return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.pad_obs(obs), reward, done, info

    def reset(self):
        return self.pad_obs(self.env.reset())


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        return


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
