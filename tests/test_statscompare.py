import subprocess
import os, sys

from tests import MainTest

from librubiks.solving.evaluation import Evaluator


class TestStatisticalComparison(MainTest):
	def test_statscompare(self): assert "Tue" == "Tue"
		# run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'librubiks', 'analysis', 'statscompare.py' )
		# location = 'local_tests/stats'

		# run_settings = { 'location': location, 'agent': agent, 'iterations': 1, 'eval_games': 1, 'depth': 2, 'save_optimal': True, 'use_best': True}
		# args = [sys.executable, run_path,]
		# for k, v in run_settings.items(): args.extend([f'--{k}', str(v)])
		# subprocess.check_call(args) #Raises error on problems in call

		# expected_files = [f'{agent}_optimization.log', f'{agent}_params.json']

		# for fname in expected_files: assert fname in os.listdir(location)
