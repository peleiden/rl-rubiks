import os, sys
import subprocess

from src.tests import MainTest


class TestRuntrain(MainTest):
	def test_run(self):
		run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rubiks', 'runtrain.py' )
		location = 'local_tests/train'
		run_settings = {'location': location, 'rollouts': 1, 'rollout_games': 2, 'rollout_depth':2, 'batch_size':2,
				'loss_weighting': 'adaptive', 'lr': 0.1, 'optim_fn': 'Adam', 'evaluations': 0, 'is2024': True, 'arch': 'fc', 'analysis': True}
		args = [sys.executable, run_path]
		for k,v in run_settings.items(): args.extend([f'--{k}', str(v)])
		subprocess.check_call(args)  # Raises error on problems in call

		#TODO Add all files
		expected_files = ['model.pt', 'train.log',  'config.json',  'model-min.pt', 'training.png', 'substate_dists.png']
		expected_train_data_files = ['rollouts.npy','losses.npy']
		print(os.listdir(location))
		print(os.listdir(os.path.join(location, 'train-data')))
		for fname in expected_files:
			assert fname in os.listdir(location)
		for fname in expected_train_data_files:
			assert fname in os.listdir(os.path.join(location, 'train-data'))




