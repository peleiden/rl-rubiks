import os, sys
import subprocess

from src.tests import MainTest


class TestRuneval(MainTest):
	def test_run(self):
		run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rubiks', 'runeval.py' )
		location = 'local_tests/eval'

		run_settings = {'location': location, 'searcher': 'BFS', 'games': 2, 'max_time': 1, 'scrambling': '2 4',
				'mcts_c': 0.6123, 'mcts_nu':.005, 'mcts_graph_search': False, 'policy_sample': True}
		args = [sys.executable, run_path]
		for k,v in run_settings.items(): args.extend([f'--{k}', str(v)])
		subprocess.check_call(args) #Raises error on problems in call

		expected_files = ['Breadth-first search_results.npy']
		for fname in expected_files:
			assert fname in os.listdir(location)


		#TODO: Test with DeepSearcher (needs a quick training first)
		# run_settings = {'location': location, 'searcher': 'MCTS', 'games': 2, 'max_time': 1, 'scrambling': '2 4',
				# 'mcts_c': 0.6123, 'mcts_nu':.005, 'mcts_graph_search': False}

