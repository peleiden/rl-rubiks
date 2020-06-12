import subprocess
import os, sys

import numpy as np
import scipy.stats as stats

from tests import MainTest


from librubiks.analysis.statscompare import StatisticalComparison
from librubiks.utils import NullLogger

class TestStatisticalComparison(MainTest):
	def test_run(self):
		location = 'local_tests'
		np.save(f'{location}/a_results.npy', np.random.randn(100))
		np.save(f'{location}/b_results.npy', np.random.randn(100))

		run_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'librubiks', 'analysis', 'statscompare.py' )
		run_settings = { 'location': location, 'alpha': 0.1}
		args = [sys.executable, run_path,]
		for k, v in run_settings.items(): args.extend([f'--{k}', str(v)])
		subprocess.check_call(args) #Raises error on problems in call

		expected_files = ['stats.log']
		for fname in expected_files: assert fname in os.listdir(location)

	def test_statcomp(self):
		A, B = np.random.randint(0, 100, 100), np.random.randint(5, 150, 100)
		p1, p2 = np.random.randint(100), np.random.randint(100)
		A[np.arange(100)[p1]] = -1
		B[np.arange(100)[p2]] = -1

		s = StatisticalComparison(None, NullLogger)
		s.names = ["a","b"]
		s.results = [A, B]

		### T test ###
		_, p_exp = stats.ttest_ind(A[A!=-1], B[B!=-1], equal_var=False)
		p_gotten, _ = s.length_ttest(0.05)
		assert np.isclose(p_exp, p_gotten)


