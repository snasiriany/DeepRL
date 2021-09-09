import os
import os.path as osp
import json
import tensorflow as tf
import csv
import math
import copy

base_options_dir = '/home/soroushn/research/options/data'
base_dir = osp.join(base_options_dir)

for env in [
	'lift', 'stack', 'pnp', 'nut_round', 'cleanup', 'wipe', 'peg_ins', 'door',
	'cleanup_twin', 'push_and_stack_twin', 'stack_twin', 'lift_twin',
	'cleanup_twin_easy', 'push_and_stack_twin_easy',
]:
	base_dir = osp.join('/home/soroushn/research/options/data', env)

	for exp_name in os.listdir(base_dir):
		tf_dir = osp.join(base_dir, exp_name, 'tf_log')

		if not os.path.exists(tf_dir):
			continue

		tf_file = os.listdir(tf_dir)[0]

		data = {}

		for e in tf.compat.v1.train.summary_iterator(os.path.join(tf_dir, tf_file)):
			for v in e.summary.value:
				if v.tag in ['episodic_return_test', 'episodic_success_test']:
					epoch = e.step / 3000
					if epoch not in data:
						data[epoch] = {
							'Epoch': epoch,
						}
					if v.tag == 'episodic_return_test':
						key = 'evalTask ac Returns Avg'
						data[epoch][key] = v.simple_value / 150.0
					elif v.tag == 'episodic_success_test':
						key = 'evalenv_infosfinalsuccess Mean'
						data[epoch][key] = v.simple_value
					else:
						raise NotImplementedError

		# copy data for epochs that we don't have data for
		epoch_list = sorted(data)
		for i in range(len(epoch_list) - 1):
			# print("loop")
			ref_epoch = epoch_list[i]
			for cur_epoch in range(math.ceil(epoch_list[i]), math.floor(epoch_list[i+1])):
				data[cur_epoch] = copy.copy(data[ref_epoch])
				data[cur_epoch]['Epoch'] = cur_epoch
		# exit()

		data_dir = osp.join(base_dir, exp_name, 'data')
		if not os.path.exists(data_dir):
			os.mkdir(data_dir)

		with open(osp.join(data_dir, 'variant.json'), 'w') as outfile:
			json.dump({}, outfile)

		with open(osp.join(data_dir, 'progress.csv'), mode='w') as csv_file:
			fieldnames = list(next(iter(data.values())).keys())
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

			writer.writeheader()
			for epoch in sorted(data):
				writer.writerow(data[epoch])

