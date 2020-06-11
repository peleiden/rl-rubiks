import os

import argparse
from glob import glob as glob #glob

import numpy as np
import scipy.stats as stats

from librubiks.utils import Logger

class StatisticalComparison:
	def __init__(self, path: str, log: Logger):
		self.p = path
		self.log = log

		self.names = None
		self.results = None

	def dataload(self):
		""" Loads data from  path self.p """
		self.names = self._check_agents(self.p)
		if not self.names:
			self.p = os.path.join(self.p, 'evaluation_results')
			self.names = self._check_agents(self.p)
			if not self.names: raise FileNotFoundError(f"No results found in {self.p} or parent folder")
		if len(self.names) > 2:
			self.log("Multiple agents were submitted")
			choices = "\n".join(f'{i}: {f}' for i, f in enumerate(self.names))
			chosen = [ int(input(f"Please choose {w} agent (give index): {choices}")) for w in ('first', 'second') ]
			self.names = [ self.names[i] for i in chosen ]
		self.results = [ np.load(os.path.join(self.p, f"{name}_results.npy")) for name in self.names ]
		self.log(f"Results loaded for agents {self.names}from path\n{self.p}")

	def length_ttest(self, alpha: float):
		""" Welch T test """
		solution_lengths = [r[r != -1] for r in self.results]
		t_obs, p = stats.ttest_ind(*solution_lengths, equal_var=False)

		self.log("Welch t-test of H0: mean(sol_lengths_agent1) = mean(sol_lengths_agent2) performed")
		self.log(f"Resulting p value and t test statistic:\n\t {p} {t_obs}")
		#TODO: Compute confidence interval
		#TODO: model validation

	def solve_proptest(self, alpha: float):
		#TODO: DO
		pass

	@staticmethod
	def _check_agents(p: str):
		# Files are named "./ASTAR (lambda=0.2, N=100) bignetwork_results.npy"
		return list({ f.split('/')[-1].split('_')[0] for f in glob(os.path.join(p, "*.npy")) }) #FIXME: Split path better


def statscompare():
	"""
	Main way to run statistical comparison by running `python librubiks/analysis/statscompare.py --help`.
	Does not support config arguments.
	"""
	parser = argparse.ArgumentParser(description='Compare two agents by doing t test of solution lengths and Xi-squared test of solve proportions')
	parser.add_argument('--location', help="Folder containing evaluation results. If exactly two different agents are contained herein,"
		"these will be compared.\nOtherwise, the user will be prompted", type =str)
	parser.add_argument('--alpha', help="Significane level used", type=float)

	args = parser.parse_args()

	comp = StatisticalComparison(args.location, Logger(os.path.join(args.location, "stats.log"), "Statistical comparison"))
	comp.dataload()
	comp.length_ttest(alpha=args.alpha)
	comp.solve_proptest(alpha=args.alpha)

if __name__ == '__main__': statscompare()
