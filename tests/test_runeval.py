import os, sys
import subprocess

from tests import MainTest
from tests.test_hyper_optim import TestOptimizer

class TestRuneval(MainTest):
	def test_run(self):
		run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  'runeval.py' )
		location = 'local_tests/eval'

		run_settings = {'location': location, 'agent': 'BFS', 'games': 2, 'max_time': 1, 'scrambling': '2 4',
				'mcts_c': 0.6123, 'mcts_graph_search': False, 'policy_sample': True}
		args = [sys.executable, run_path,]
		for k,v in run_settings.items(): args.extend([f'--{k}', str(v)])
		subprocess.check_call(args) #Raises error on problems in call

		expected_files = ['Breadth-first search_results.npy', 'Breadth-first search_states_seen.npy' , 'eval_sollengths.png', 'eval_winrates.png']

		for fname in expected_files:
			assert fname in os.listdir(location)


		# DeepAgent + Optimization hyper parameter test
		to = TestOptimizer()
		location = to.test_agent_optim(['AStar'])

		dank_unlikely_number = 0.6969
		run_settings = {'location': location, 'agent': 'AStar', 'games': 1, 'max_time': 1, 'scrambling': '1 3',
				'astar_lambda':  dank_unlikely_number, 'optimized_params' : True}
		args = [sys.executable, run_path,]
		for k,v in run_settings.items(): args.extend([f'--{k}', str(v)])
		subprocess.check_call(args) #Raises error on problems in call

		expected_files = ['eval_sollengths.png', 'eval_winrates.png']

		file_list = os.listdir(location)
		for fname in expected_files:
			assert fname in file_list

		astar_found = False
		for found_file in file_list: #n**2 + n goes brrr
			if 'AStar' in found_file:
				astar_found = True
				assert str(dank_unlikely_number) not in found_file, "To test whether the optimized param was used"
		assert astar_found, "Find output file"
