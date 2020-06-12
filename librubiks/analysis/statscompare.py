import os

import argparse
from glob import glob as glob #glob
from ast import literal_eval
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from librubiks.utils import Logger

class StatisticalComparison:
	def __init__(self, path: str, log: Logger, compare_all: bool = False):
		self.p = path
		self.log = log
		self.compare_all = compare_all

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
			if not self.compare_all:
				self.log("Multiple agents were submitted. If you want to run all combinations, rerun with --compare_all True.")
				choices = "\n".join(f'{i}: {f}' for i, f in enumerate(self.names))
				chosen = [ int(input(f"Please choose {w} agent (give index): {choices}")) for w in ('first', 'second') ]
				self.names = [ self.names[i] for i in chosen ]
		self.results = [ np.load(os.path.join(self.p, f"{name}_results.npy")) for name in self.names ]
		self.log(f"Results loaded for agents\n\t{self.names}\nfrom path\n\t{self.p}")

	def run_comparisons(self, alpha: float):
		""" Do all the statistical comparisons of agent combinations"""
		length_ps, solution_ps , all_names = list(), list(), list()
		for idcs in combinations(range(len(self.names)), r=2):
			names = [self.names[i] for i in idcs]
			results = [self.results[i] for i in idcs]
			all_names.append(names)
			self.log.section(f"Comparison of agents\n\t{names}")
			p, _ = self.length_ttest(results, alpha)
			length_ps.append(p)
			p, _ = self.solve_proptest(results, alpha)
			solution_ps.append(p)

		length_ps, solution_ps = self.fdr_correction(np.array(length_ps)), self.fdr_correction(np.array(solution_ps))
		self.log.section("CORRECTED p values")
		for i, name in enumerate(all_names):
			self.log(f"Corrected p values for {name}")
			self.log(f"Corrected solution length p value for {length_ps[i]}", with_timestamp=False)
			self.log(f"Corrected solution proportion p value for {solution_ps[i]}", with_timestamp=False)
		return length_ps, solution_ps

	def length_ttest(self, results: list, alpha: float):
		""" Welch T test that solution lengths are equal.
		See method 3.47 and 3.49 in https://02402.compute.dtu.dk/enotes/ """
		self.log.section("Test of equal solution lengths")
		solution_lengths = [r[r != -1] for r in results]
		V = np.array([s.var(ddof=1) for s in solution_lengths])
		M = np.array([s.mean() for s in solution_lengths])
		N = np.array([s.shape[0] for s in solution_lengths])
		mu = M[0]-M[1]
		m_var = (V/N).sum() #V[mean(X)-mean(Y)]
		df_welch = m_var**2 / ( (V[0]/N[0])**2/(N[0]-1) + (V[1]/N[1])**2/(N[1]-1))

		t_obs = (mu)/np.sqrt(m_var)
		p = 2*(1-stats.t.cdf(abs(t_obs), df=df_welch))
		qt = stats.t.ppf(1-alpha/2, df=df_welch)
		mean_error = qt * np.sqrt(m_var)
		CI = mu + np.array([-1, 1]) * mean_error
		self.log("Two-sided Welch t-test of H0: mean(sol_lengths_agent1) = mean(sol_lengths_agent2) performed\n"
			f"in t-distribution with {df_welch} degrees of freedom", with_timestamp=False)
		self.log(f"Resulting (non-corrected) p value and t test statistic:\n\t {p} {t_obs}", with_timestamp=False)
		self.log(f"Confidence interval at level {alpha} of difference is\n\t{mu} +/- {mean_error}\n\t(which is {CI})",
				with_timestamp=False)
		return p, CI

	def solve_proptest(self, results: list, alpha: float):
		""" Test that solve proportions are equal.
		See method 7.18 in https://02402.compute.dtu.dk/enotes/ """
		self.log.section("Test of equal solve proportions")
		X = np.array([(r != 1).sum() for r in results])
		N = np.array([r.size for r in results])
		P = X / N
		mu = P[0] - P[1]
		prop = X.sum() / N.sum()
		if mu == 0:
			if P[0] == 1:
				self.log("Proportions are both at 100%, no analysis can be carried out", with_timestamp=False)
				return 0, np.array([0,0])
			if P[1] == 0:
				self.log("Proportions are both at 0%, no analysis can be carried out", with_timestamp=False)
				return 0, np.array([0,0])
		z_obs = mu / np.sqrt( prop * (1-prop) * (1/N).sum() )
		p = 2*(1-stats.norm.cdf(abs(z_obs)))

		qz = stats.norm.ppf(1-alpha/2)
		mean_error = qz * np.sqrt((P*(1-P)/N).sum())
		CI = mu + np.array([-1, 1]) * mean_error
		self.log("Two-sided proportion test of H0: mean(sol_prop) = mean(sol_prop) performed\n"
			f"in the standard normal distribution", with_timestamp=False)
		self.log(f"Resulting (non-corrected) p value and z test statistic:\n\t {p} {z_obs}", with_timestamp=False)
		self.log(f"Confidence interval at level {alpha} of difference is\n\t{mu} +/- {mean_error}\n\t(which is {CI})",
				with_timestamp=False)
		self.log("Proportion samples (all should be > 10 for accurate model): "
				f"{[int(i) for i in N*P]}, {[int(i) for i in N*(1-P)]}", with_timestamp=False)
		return p, CI

	def normality_plot(self, k: int = 10000):
		"""Check normality of solution lengths"""
		for i, result in enumerate(self.results):
			result, name = result[result!=-1], self.names[i]
			plt.figure(figsize=(10,10))

			plt.subplot(2,2,1)
			Z = (result-result.mean())/(result.std(ddof=1) + 1e-6)
			stats.probplot(Z, dist="norm", plot=plt)
			plt.title("QQplot of data")

			plt.subplot(2,2,2)
			plt.title(f"Histogram: {result.size} data points")
			plt.xlabel("Solution lengths")
			plt.hist(result, density=True, align="left", edgecolor="black")
			x = np.linspace(*plt.xlim(), 1000)
			p = stats.norm.pdf(x, result.mean(), result.std())
			plt.plot(x, p, linewidth=2)

			means = np.array(self.bootstrap_means(result, k))
			plt.subplot(2,2,3)
			Z = (means-means.mean())/(means.std(ddof=1) + 1e-6)
			stats.probplot(Z, dist="norm", plot=plt)
			plt.title("QQplot of bootstrapped means")

			plt.subplot(2,2,4)
			plt.title(f"Histogram of {k} boostrapped means")
			plt.xlabel("Mean solution lengths")
			plt.hist(means, density=True, align="left", edgecolor="black", bins = max(50, k//500))
			x = np.linspace(*plt.xlim(), 1000)
			p = stats.norm.pdf(x, means.mean(), means.std())
			plt.plot(x, p, linewidth=2)

			plt.suptitle(f"Normality for {name}")
			plt.tight_layout()
			plt.subplots_adjust(top=0.88)
			plt.savefig(os.path.join(self.p, f"{name}_normality.png"))
			self.log(f"Normality plot saved for {name}")

	@staticmethod
	def bootstrap_means(data: np.ndarray, k: int):
		""" Boostrap means to check for mean distribution """
		l = data.size
		return [data[np.random.randint(0, l-1, l)].mean() for _ in range(k)]

	@staticmethod
	def fdr_correction(p_vals: np.ndarray):
		""" Implement the False Discovery Rate/Benjamini-Hochberg correction of p values.
		see http://www.biostathandbook.com/multiplecomparisons.html """
		ranked_p = stats.rankdata(p_vals)
		fdr = p_vals * len(p_vals) / ranked_p
		fdr[fdr > 1] = 1
		return fdr

	@staticmethod
	def _check_agents(p: str):
		# Files are named "evaluation_results/ASTAR (lambda=0.2, N=100) bignetwork_results.npy"
		return list({ f.split('/')[-1].split('_')[0] for f in glob(os.path.join(p, "*.npy")) }) #FIXME: Split path better


def statscompare():
	"""
	Main way to run statistical comparison by running `python librubiks/analysis/statscompare.py --help`.
	Does not support config arguments.
	"""
	parser = argparse.ArgumentParser(description='Compare two agents by doing t test of solution lengths and Xi-squared test of solve proportions')
	parser.add_argument('--location', help="Folder containing evaluation results. If exactly two different agents are contained herein,"
		"these will be compared.\nOtherwise, the user will be prompted", type =str)
	parser.add_argument('--alpha', help="Significane level used", type=float, default=0.01)
	parser.add_argument('--compare_all', help="If true, all comparisons in folder is run, using p value cprrection",
		type=literal_eval, default=True, choices=[True, False])

	args = parser.parse_args()

	comp = StatisticalComparison(args.location, Logger(os.path.join(args.location, "stats.log"), "Statistical comparison"), compare_all = args.compare_all)
	comp.dataload()
	comp.run_comparisons(alpha=args.alpha)
	comp.normality_plot()

if __name__ == '__main__': statscompare()
