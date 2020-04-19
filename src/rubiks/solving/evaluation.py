import os
from os import cpu_count

import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 22})

from src.rubiks.solving.search import RandomDFS, BFS, PolicySearch, MCTS
from src.rubiks.utils.logger import NullLogger, Logger
from src.rubiks.utils.ticktock import TickTock

from src.rubiks.cube.cube import Cube
from src.rubiks.solving.agents import Agent

# Multiprocessing is silly, so all functions have to be top-level
# This also means all info has to be parsed in with a single argument
# https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
def _eval_game(cfg: (Agent, int, int)):
	agent, max_time, depth = cfg
	turns_to_complete = -1  # -1 for unfinished
	state, _, _ = Cube.scramble(depth, True)
	solution_found, n_actions = agent.generate_action_queue(state, max_time)
	if solution_found:
		turns_to_complete = n_actions
	return turns_to_complete


class Evaluator:
	def __init__(self,
			n_games			= 420,  # Nice
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

	def eval(self, agent: Agent, max_threads=None):
		"""
		Evaluates an agent
		"""
		self.log.section(f"Evaluation of {agent}")
		self.log(f"{self.n_games*len(self.scrambling_depths)} games with max time per game {self.max_time}\nExpected time <~ {self.approximate_time()/60:.2f} min")

		# Builds configurations for runs
		cfgs = []
		for i, d in enumerate(self.scrambling_depths):
			for _ in range(self.n_games):
				cfgs.append((agent, self.max_time, d))

		max_threads = max_threads or cpu_count()
		threads = min(max_threads, cpu_count()) if agent.allow_mt() else 1
		if agent.allow_mt():
			self.log(f"Evaluating {agent} on {threads} threads")
			with mp.Pool(threads) as p:
				res = p.map(_eval_game, cfgs)
		else:
			res = []
			for i, cfg in enumerate(cfgs):
				self.tt.profile(f"Evaluation of {agent}. Depth {cfg[2]}")
				res.append(_eval_game(cfg))
				self.log.verbose(f"Performing evaluation {i+1} / {len(cfgs)}. Depth: {cfg[2]}. Explored states: {len(agent)}")
				self.tt.end_profile(f"Evaluation of {agent}. Depth {cfg[2]}")
		res = np.reshape(res, (len(self.scrambling_depths), self.n_games))

		self.log(f"Evaluation results")
		for i, d in enumerate(self.scrambling_depths):
			self.log(f"Scrambling depth {d}", with_timestamp=False)
			self.log(f"\tShare completed: {np.count_nonzero(res[i]!=-1)*100/len(res[i]):.2f} %", with_timestamp=False)
			if (res[i]!=-1).any():
				self.log(f"\tMean turns to complete (ex. unfinished): {res[i][res[i]!=-1].mean():.2f}", with_timestamp=False)
				self.log(f"\tMedian turns to complete (ex. unfinished): {np.median(res[i][res[i]!=-1]):.2f}", with_timestamp=False)
		self.log.verbose(f"Evaluation runtime\n{self.tt}")

		return res

	def approximate_time(self):
		return self.max_time*self.n_games*len(self.scrambling_depths)

	def plot_this_eval(self, eval_results: dict, save_dir: str,  **kwargs):
		self.log("Creating plot of evaluation")
		settings = {
			'n_games': self.n_games,
			'max_time': self.max_time,
			'scrambling_depths': self.scrambling_depths
		}
		self.plot_an_eval(eval_results, save_dir, settings, **kwargs)

	@staticmethod
	def plot_an_eval(eval_results: dict, save_dir: str,  eval_settings: dict, show: bool=False, title: str=''):
		"""
		{agent: results from self.eval}
		"""
		#depth, win%-graph
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		ax.set_ylabel(f"Percentage of {eval_settings['n_games']} games won")
		ax.set_xlabel(f"Scrambling depth: Number of random rotations applied to cubes")
		ax.locator_params(axis='x', integer=True, tight=True)

		cmap = plt.get_cmap('gist_rainbow')
		colors = [cmap(i) for i in np.linspace(0, 1, len(eval_results))]

		for i, (agent, results) in enumerate(eval_results.items()):
			color = colors[i]
			win_percentages = (results != -1).mean(axis=1) * 100

			ax.plot(eval_settings['scrambling_depths'], win_percentages, linestyle='dashdot', color=color)
			ax.scatter(eval_settings['scrambling_depths'], win_percentages, color=color, label=f"Win % of {agent}")
		ax.legend()
		fig.tight_layout()
		ax.grid(True)
		ax.set_title(title if title else f"Cubes solved in {eval_settings['max_time']:.2f} seconds")

		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "eval_winrates.png")
		plt.savefig(path)

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

		if show: plt.show()
		plt.clf()
