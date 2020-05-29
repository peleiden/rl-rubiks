import subprocess
import os, sys

from tests import MainTest

from librubiks.solving.hyper_optim import Optimizer, BayesianOptimizer
from librubiks.model import ModelConfig, Model
class TestOptimizer(MainTest):
	# def test_BO(self): #TODO: Find out why this fails with weird arraycheck error in GH actions
		# def f(params): return -params['x']+42
#
		# bo = BayesianOptimizer(f, {'x': (0, 42)},  n_restarts=2)
		# bo.optimize(2)
		# assert len(bo.score_history) == 2
		# assert bo.highscore >= bo.score_history[0]

	def test_searcher_optim(self, searchers = ['MCTS', 'AStar']):

		run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'librubiks', 'solving', 'hyper_optim.py' )
		location = 'local_tests/optim'

		net = Model(ModelConfig())
		net.save(location)
		for searcher in searchers:

			run_settings = { 'location': location, 'searcher': searcher, 'iterations': 1, 'eval_games': 1, 'depth': 2, 'save_optimal': True, 'use_best': True}
			args = [sys.executable, run_path,]
			for k,v in run_settings.items(): args.extend([f'--{k}', str(v)])
			subprocess.check_call(args) #Raises error on problems in call

			expected_files = [f'{searcher}_optimization.log', f'{searcher}_params.json']

			for fname in expected_files: assert fname in os.listdir(location)

		return location
