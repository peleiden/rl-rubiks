import subprocess
import os, sys

from tests import MainTest

from librubiks.solving.hyper_optim import Optimizer, BayesianOptimizer, MCTS_optimize
from librubiks.model import ModelConfig, Model
class TestOptimizer(MainTest):
	def test_BO(self):
		def f(params): return -params['x']+42

		bo = BayesianOptimizer(f, {'x': (0, 42)},  n_restarts=1)
		bo.optimize(2)
		assert len(bo.score_history) == 2
		assert bo.highscore >= bo.score_history[0]

	def test_MCTS_optim(self):
		run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'librubiks', 'solving', 'hyper_optim.py' )
		location = 'local_tests/optim'

		net = Model(ModelConfig())
		net.save(location)

		run_settings = {'location': location, 'iterations': 1, 'eval_games': 1, 'policy_type': 'w' }
		args = [sys.executable, run_path,]
		for k,v in run_settings.items(): args.extend([f'--{k}', str(v)])
		subprocess.check_call(args) #Raises error on problems in call

		expected_files = ['optimizer.log']

		for fname in expected_files:
			assert fname in os.listdir(location)