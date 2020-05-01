import os

import numpy as np
import matplotlib.colors as mcolour
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 22})

from src.rubiks.utils.logger import NullLogger, Logger
from src.rubiks.utils.ticktock import TickTock

from src.rubiks.cube.cube import Cube
from src.rubiks.solving.agents import Agent


class Evaluator:
	def __init__(self,
			n_games				= 420,  # Nice
			max_time			= 600,  # Max time to completion per game
			scrambling_depths	= range(1, 10),
			logger: Logger		= NullLogger()
		):

		self.n_games = n_games
		self.max_time = max_time

		self.tt = TickTock()
		self.log = logger
		self.scrambling_depths = np.array(scrambling_depths)

		self.log("\n".join([
			"Creating evaluator",
			f"Games per scrambling depth: {self.n_games}",
			f"Scrambling depths: {scrambling_depths}",
		]))

	def approximate_time(self):
		return self.max_time * self.n_games * len(self.scrambling_depths)

	def _eval_game(self, agent: Agent, depth: int):
		turns_to_complete = -1  # -1 for unfinished
		state, _, _ = Cube.scramble(depth, True)
		solution_found, n_actions = agent.generate_action_queue(state, self.max_time)
		if solution_found:
			turns_to_complete = n_actions
		return turns_to_complete

	def eval(self, agent: Agent):
		"""
		Evaluates an agent
		Returns results which is an a len(self.scrambling_depths) x self.n_games matrix
		Each entry contains the number of steps needed to solve the scrambled cube or -1 if not solved
		"""
		self.log.section(f"Evaluation of {agent}")
		self.log(f"{self.n_games*len(self.scrambling_depths)} games with max time per game {self.max_time}\nExpected time <~ {self.approximate_time()/60:.2f} min")

		# Builds configurations for runs
		cfgs = []
		for i, d in enumerate(self.scrambling_depths):
			for _ in range(self.n_games):
				cfgs.append((agent, self.max_time, d))

		res = []
		for i, cfg in enumerate(cfgs):
			self.tt.profile(f"Evaluation of {agent}. Depth {cfg[2]}")
			res.append(self._eval_game(agent, cfg[2]))
			self.log.verbose(f"Performing evaluation {i+1} / {len(cfgs)}. Depth: {cfg[2]}. Explored states: {len(agent)}")
			self.tt.end_profile(f"Evaluation of {agent}. Depth {cfg[2]}")
		res = np.reshape(res, (len(self.scrambling_depths), self.n_games))

		self.log(f"Evaluation results")
		for i, d in enumerate(self.scrambling_depths):
			share_completed = np.count_nonzero(res[i]!=-1)*100/len(res[i])
			mean_turns = res[i][res[i]!=-1].mean()
			median_turns = np.median(res[i][res[i]!=-1])
			self.log(f"Scrambling depth {d}", with_timestamp=False)
			self.log(f"\tShare completed: {share_completed:.2f} %", with_timestamp=False)
			if (res[i]!=-1).any():
				self.log(f"\tMean turns to complete (ex. unfinished): {mean_turns:.2f}", with_timestamp=False)
				self.log(f"\tMedian turns to complete (ex. unfinished): {median_turns:.2f}", with_timestamp=False)
		self.log(f"Sum score: {self.sum_score(res).mean():.2f}", with_timestamp=False)
		self.log.verbose(f"Evaluation runtime\n{self.tt}")

		return res

	def sum_score(self, res: np.ndarray) -> np.ndarray:
		"""
		Computes sum score game wise, that is it returns an array of length self.n_games
		It assumes that all srambling depths lower that self.scrambling_depths[0] are always solved
		and that all depths above self.scrambling_depths[-1] are never solved
		Overall sum_score is the mean of the returned array
		:param res: Numpy array of evaluation results as returned by self.eval
		:return: Numpy array of length self.n_games
		"""
		solved = res != -1
		lower_depths = self.scrambling_depths[0] - 1
		return solved.sum(axis=0) + lower_depths

	def plot_this_eval(self, eval_results: dict, save_dir: str,  **kwargs):
		self.log("Creating plot of evaluation")
		settings = {
			'n_games': self.n_games,
			'max_time': self.max_time,
			'scrambling_depths': self.scrambling_depths
		}
		save_paths = self.plot_an_eval(eval_results, save_dir, settings, **kwargs)
		self.log(f"Saved evaluation plots to {save_paths}")

	def plot_an_eval(self, eval_results: dict, save_dir: str,  eval_settings: dict, show: bool=False, title: str=''):
		"""
		{agent: results from self.eval}
		"""
		save_paths = []
		#depth, win%-graph
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		ax.set_ylabel(f"Percentage of {eval_settings['n_games']} games won")
		ax.set_xlabel(f"Scrambling depth: Number of random rotations applied to cubes")
		ax.locator_params(axis='x', integer=True, tight=True)

		tab_colours = list(mcolour.TABLEAU_COLORS)
		colours = [tab_colours[i%len(tab_colours)] for i in range(len(eval_results))]

		for i, (agent, results) in enumerate(eval_results.items()):
			color = colours[i]
			win_percentages = (results != -1).mean(axis=1) * 100

			ax.plot(eval_settings['scrambling_depths'], win_percentages, linestyle='dashdot', color=color)
			ax.scatter(eval_settings['scrambling_depths'], win_percentages, color=color, label=f"Win % of {agent}")
		ax.legend()
		ax.set_ylim([-5, 105])
		ax.grid(True)
		ax.set_title(title if title else f"Cubes solved in {eval_settings['max_time']:.2f} seconds")
		fig.tight_layout()

		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "eval_winrates.png")
		plt.savefig(path)
		save_paths.append(path)

		if show: plt.show()
		plt.clf()

		# solution length boxplots

		fig, axes = plt.subplots(len(eval_results), 1, figsize=(19.2, 10.8))

		for i, (agent, results) in enumerate(eval_results.items()):

			ax = axes[i] if len(eval_results) > 1 else axes
			ax.set_title(f'Solution lengths for {agent} in {eval_settings["max_time"]:.2f} s')

			ax.set_ylabel(f"Solution length")
			ax.set_xlabel(f"Scrambling depth")

			#Handling that some might not even win any games
			plotables = (results != -1).any(axis=1)
			results = [depth[depth != -1] for depth in results[plotables]]
			depths = [eval_settings['scrambling_depths'][i] for i  in range(len(plotables)) if plotables[i]]
			if len(depths): ax.boxplot(results, labels=depths)
			ax.grid(True)

		fig.tight_layout()
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "eval_sollengths.png")
		plt.savefig(path)
		save_paths.append(path)

		if show: plt.show()
		plt.clf()

		# Histograms of sum scores
		normal_pdf = lambda x, mu, sigma: np.exp(-1/2 * ((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		sss = np.array([self.sum_score(results) for results in eval_results.values()])
		mus, stds = [ss.mean() for ss in sss], [ss.std() for ss in sss]
		lower, higher = sss.min() - 2, sss.max() + 2
		bins = np.arange(lower, higher+1)
		ax.hist(x		= sss.T,
				bins	= bins,
				density	= True,
				color	= colours,
				align	= "left",
				label	= [f"{agent}. mu = {mus[i]:.2f}" for i, agent in enumerate(eval_results.keys())])
		for i in range(len(eval_results)):
			x = np.linspace(lower, higher, 100)
			y = normal_pdf(x, mus[i], stds[i])
			x = x[~np.isnan(y)]
			y = y[~np.isnan(y)]
			plt.plot(x, y, color="black")
		ax.set_xlim([lower, higher])
		ax.set_xticks(bins)
		ax.set_title(f"Single game S distributions (evaluated on {self.max_time:.2f} s per game)")
		ax.set_xlabel("S")
		ax.set_ylabel("Frequency")
		ax.legend(loc=1)
		path = os.path.join(save_dir, "eval_sum_scores.png")
		plt.savefig(path)
		save_paths.append(path)

		if show: plt.show()
		plt.clf()

		return save_paths
