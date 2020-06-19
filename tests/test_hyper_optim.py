import subprocess
import os, sys

import numpy as np

from tests import MainTest
from librubiks.solving.hyper_optim import Optimizer, BayesianOptimizer, GridSearch
from librubiks.model import ModelConfig, Model


class TestOptimizer(MainTest):
	
	@staticmethod
	def f(params):
		return -params['x'] + 42, np.empty(5), np.empty(5)
	
	def test_GS(self):
		gs = GridSearch(self.f, {'x': (0, 42)})
		gs.optimize(2) # WARNING: fails on sklearn 0.23
		assert len(gs.score_history) == 2
		assert gs.highscore >= gs.score_history[1]
	
	def test_BO(self):
		"""NOTE: This fails on scikit-learn 0.23, see https://github.com/fmfn/BayesianOptimization/issues/231"""
#
		bo = BayesianOptimizer(self.f, {'x': (0, 42)},  n_restarts=2)
		bo.optimize(2)  # WARNING: fails on scikit-learn 0.23
		assert len(bo.score_history) == 2
		assert bo.highscore >= bo.score_history[1]

	def test_agent_optim(self, agents=['MCTS', 'AStar', 'EGVM']):

		run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'librubiks', 'solving', 'hyper_optim.py' )
		location = 'local_tests/optim'

		net = Model(ModelConfig())
		net.save(location)
		for agent in agents:

			run_settings = { 'location': location, 'agent': agent, 'iterations': 1, 'eval_games': 1, 'depth': 2, 'save_optimal': True, 'use_best': True, 'optimizer': 'BO' }
			args = [sys.executable, run_path,]
			for k, v in run_settings.items(): args.extend([f'--{k}', str(v)])
			subprocess.check_call(args)  # Raises error on problems in call

			expected_files = [f'{agent}_optimization.log', f'{agent}_params.json']

			for fname in expected_files: assert fname in os.listdir(location)

		return location
