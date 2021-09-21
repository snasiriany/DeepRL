from deep_rl import *
import subprocess
import argparse
import os.path as osp
from rlkit.launchers.visualization import get_image, annotate_image

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
	agent = ASquaredCPPOAgent(config)

	agent.all_options = []
	saved = '{}/model/{}'.format(config.ckpt, config.step)
	agent.load(saved)

	_, _, _, path = agent.eval_episode(image_obs_in_info=True)
	dump_video(
		path,
		save_path=osp.join(agent.logger.log_dir, 'video.mp4'),
		horizon=config.horizon,
	)


def dump_video(
		path,
		save_path,
		pad_length=15,
		pad_color=0,
		do_timer=True,
		horizon=100,
		imsize=512,
		num_channels=3,
):
	import os
	import skvideo.io
	import time

	# logdir = logger.get_snapshot_dir()

	frames = []
	H = imsize
	W = imsize

	columns = 1
	rows = 1
	N = rows * columns

	rollout_actions = []
	successes = []

	for i in range(N):
		start = time.time()

		rollout_actions.append(path['actions'])
		successes.append([info.get('success', False) for info in path['env_infos']])

		l = []
		for j in range(len(path['env_infos'])):
			imgs = path['env_infos'][j]['image_obs']
			# ac_str = 'Atomic'
			ac_str = 'Option {}'.format(path['options'][j][0] + 1)

			success = successes[i][max(j-1, 0)]

			for img in imgs:
				img = np.flip(img, axis=0)
				img[-80:,:,:] = 235
				img = get_image(
					img,
					pad_length=pad_length,
					pad_color=(0, 225, 0) if success else pad_color,
					imsize=imsize,
				)
				if success:
					ac_str = 'Success'
				img = annotate_image(
					img,
					text=ac_str,
					imsize=imsize,
					color=(0, 175, 0) if success else (0,0,0,),
					loc='ll',
				)
				l.append(img)

			if success:
				break

		frames.append(l)

		if do_timer:
			print(time.time() - start)

	for i in range(len(frames)):
		last_img = frames[i][-1]
		for _ in range(horizon - len(frames[i])):
			frames[i].append(last_img)

	frames = np.array(frames, dtype=np.uint8)
	path_length = frames.size // (
			N * (H + 2 * pad_length) * (W + 2 * pad_length) * num_channels
	)
	frames = np.array(frames, dtype=np.uint8).reshape(
		(N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
	)
	f1 = []
	for k1 in range(columns):
		f2 = []
		for k2 in range(rows):
			k = k1 * rows + k2
			f2.append(frames[k:k + 1, :, :, :, :].reshape(
				(path_length, H + 2 * pad_length, W + 2 * pad_length,
				 num_channels)
			))
		f1.append(np.concatenate(f2, axis=1))
	outputdata = np.concatenate(f1, axis=2)

	skvideo.io.vwrite(save_path, outputdata)
	print("Saved video to ", save_path)


if __name__ == '__main__':
	# noinspection PyTypeChecker
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='HalfCheetah-v2')
	parser.add_argument('--label', type=str, default='eval')
	# parser.add_argument('--num_seeds', type=int, default=1)
	# parser.add_argument('--no_video', action='store_true')
	parser.add_argument('--no_gpu', action='store_true')
	parser.add_argument('--gpu_id', type=int, default=-1)
	# parser.add_argument('--debug', action='store_true')

	args = parser.parse_args()

	if args.env == 'lift':
		ckpts = [
			'/home/soroushn/research/options/data/lift/lift-dac-train1-210824-101339',
			'/home/soroushn/research/options/data/lift/lift-dac-train1-210824-101408',
			'/home/soroushn/research/options/data/lift/lift-dac-train1-210824-101556',
			'/home/soroushn/research/options/data/lift/lift-dac-train1-210824-101719',
		]
		step = 4966400
		horizon = 75
	elif args.env == 'pnp':
		ckpts = [
			'/home/soroushn/research/options/data/pnp/pnp-dac-train1-210824-101656',
			'/home/soroushn/research/options/data/pnp/pnp-dac-train1-210824-101828',
			'/home/soroushn/research/options/data/pnp/pnp-dac-train1-210824-102039',
			'/home/soroushn/research/options/data/pnp/pnp-dac-train1-210824-102052',
			'/home/soroushn/research/options/data/pnp/pnp-dac-train1-210824-102109',
		]
		step = 4966400
		horizon = 75
	elif args.env == 'cleanup':
		ckpts = [
			'/home/soroushn/research/options/data/cleanup/cleanup-dac-train1-210824-102430',
			'/home/soroushn/research/options/data/cleanup/cleanup-dac-train1-210824-102444',
			'/home/soroushn/research/options/data/cleanup/cleanup-dac-train1-210824-102500',
			'/home/soroushn/research/options/data/cleanup/cleanup-dac-train1-210824-102516',
			'/home/soroushn/research/options/data/cleanup/cleanup-dac-train1-210824-102539',
		]
		step = 9932800
		horizon = 125
	elif args.env == 'nut_round':
		ckpts = [
			'/home/soroushn/research/options/data/nut_round/nut_round-dac-train1-210824-102259',
			'/home/soroushn/research/options/data/nut_round/nut_round-dac-train1-210824-102314',
			'/home/soroushn/research/options/data/nut_round/nut_round-dac-train1-210824-102327',
			'/home/soroushn/research/options/data/nut_round/nut_round-dac-train1-210824-102345',
			'/home/soroushn/research/options/data/nut_round/nut_round-dac-train1-210824-102353',
		]
		step = 9932800
		horizon = 75
	elif args.env == 'peg_ins':
		ckpts = [
			'/home/soroushn/research/options/data/peg_ins/peg_ins-dac-train1-210824-102531',
			'/home/soroushn/research/options/data/peg_ins/peg_ins-dac-train1-210824-102621',
			'/home/soroushn/research/options/data/peg_ins/peg_ins-dac-train1-210824-102634',
			'/home/soroushn/research/options/data/peg_ins/peg_ins-dac-train1-210824-102645',
			'/home/soroushn/research/options/data/peg_ins/peg_ins-dac-train1-210824-102655',
		]
		step = 15052800
		horizon = 75
	else:
		raise NotImplementedError


	for ckpt in ckpts:
		mkdir('log')
		mkdir('data')
		random_seed()
		set_one_thread()
		select_device(args.gpu_id)

		a_squared_c_ppo_continuous(
			game=args.env,
			learning='all',
			log_level=1,
			num_o=4,
			opt_ep=5,
			freeze_v=False,

			tag='{}-dac-{}'.format(args.env, args.label),
			ckpt=ckpt,
			step=step,
			horizon=horizon,
		)