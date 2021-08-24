from deep_rl import *
import subprocess
import argparse


# PPO
def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('gate', nn.Tanh())
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 3e7)
    config = Config()
    config.merge(kwargs)

    # if config.tasks:
    #     set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOAgent(config))


# DAC+PPO
def a_squared_c_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.Tanh())
    kwargs.setdefault('freeze_v', False)
    kwargs.setdefault('opt_ep', 5)
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 3e7)
    kwargs.setdefault('beta_weight', 0)
    config = Config()
    config.merge(kwargs)

    # if config.tasks:
    #     set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = config.opt_ep
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    run_steps(ASquaredCPPOAgent(config))


# OC
def oc_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.Tanh())
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 3e7)
    kwargs.setdefault('num_workers', 16)
    config = Config()
    config.merge(kwargs)

    # if config.tasks:
    #     set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.random_option_prob = LinearSchedule(0.1)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.gradient_clip = 0.5
    config.rollout_length = 5
    config.beta_reg = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.target_network_update_freq = int(1e3)
    run_steps(OCAgent(config))


# PPOC
def ppoc_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('gate', nn.Tanh())
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 3e7)
    config = Config()
    config.merge(kwargs)

    # if config.tasks:
    #     set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = Task(config.game)

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.beta_reg = 0.01
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOCAgent(config))


if __name__ == '__main__':
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--label', type=str, default='test')
    # parser.add_argument('--num_seeds', type=int, default=1)
    # parser.add_argument('--no_video', action='store_true')
    # parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=-1)
    # parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    select_device(args.gpu_id)

    if args.env in ['lift', 'door', 'pnp']:
        save_interval=2048*25
    elif args.env in ['wipe']:
        save_interval=2048*10
    else:
        save_interval=2048*50


    a_squared_c_ppo_continuous(
        game=args.env,
        learning='all',
        log_level=1,
        num_o=4,
        opt_ep=5,
        freeze_v=False,

        save_interval=save_interval,
        eval_interval=2048*10,
        eval_episodes=20,
        tag='{}-dac-{}'.format(args.env, args.label),
    )

    # ppo_continuous(
    #     game=args.env,
    #     log_level=1,
    #
    #     eval_interval=2048*3,
    #     tag='{}-ppo-{}'.format(args.env, args.label),
    # )

    # oc_continuous(
    #     game=args.env,
    #     log_level=1,
    #     num_o=4,
    #
    #     tag='{}-oc-{}'.format(args.env, args.label),
    # )

    # ppoc_continuous(
    #     game=args.env,
    #     log_level=1,
    #     num_o=4,
    #
    #     tag='{}-ppoc-{}'.format(args.env, args.label),
    # )
