import os

import numpy as np
from scipy import stats
import matplotlib.colors as mcolour
import matplotlib.pyplot as plt

from librubiks import cube, rc_params, rc_params_small
from librubiks.utils import NullLogger, Logger, TickTock, TimeUnit, bernoulli_error
from librubiks.solving import agents

plt.rcParams.update(rc_params)


class Evaluator:
	def __init__(self,
		         n_games,
		         scrambling_depths: range or list,
		         max_time = None,  # Max time to completion per game
		         max_states = None,  # The max number of states to explore per game
		         logger: Logger = NullLogger()
		):

		self.n_games = n_games
		self.max_time = max_time
		self.max_states = max_states

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

	def _eval_game(self, agent: agents.Agent, depth: int):
		turns_to_complete = -1  # -1 for unfinished
		state, _, _ = cube.scramble(depth, True)
		solution_found = agent.search(state, self.max_time, self.max_states)
		if solution_found: turns_to_complete = len(agent.action_queue)
		return turns_to_complete

	def eval(self, agent: agents.Agent) -> (np.ndarray, np.ndarray, np.ndarray):
		"""
		Evaluates an agent
		Returns results which is an a len(self.scrambling_depths) x self.n_games matrix
		Each entry contains the number of steps needed to solve the scrambled cube or -1 if not solved
		"""
		self.log.section(f"Evaluation of {agent}")
		self.log("\n".join([
			f"{self.n_games*len(self.scrambling_depths)} cubes",
			f"Maximum solve time per cube is {TickTock.stringify_time(self.max_time, TimeUnit.second)} "
			f"and estimated total time <= {TickTock.stringify_time(self.approximate_time(), TimeUnit.minute)}" if self.max_time else "No time limit given",
			f"Maximum number of explored states is {TickTock.thousand_seps(self.max_states)}" if self.max_states else "No max states given",
		]))
		
		res = []
		states = []
		times = []
		for d in self.scrambling_depths:
			for _ in range(self.n_games):
				self.tt.profile(f"Evaluation of {agent}. Depth {d}")
				r = self._eval_game(agent, d)
				t = self.tt.end_profile(f"Evaluation of {agent}. Depth {d}")

				res.append(r)
				states.append(len(agent))
				times.append(t)
			self.log.verbose(f"Performed evaluation at depth: {d}/{self.scrambling_depths[-1]}")

		res = np.reshape(res, (len(self.scrambling_depths), self.n_games))
		states = np.reshape(states, (len(self.scrambling_depths), self.n_games))
		times = np.reshape(times, (len(self.scrambling_depths), self.n_games))

		self.log(f"Evaluation results")
		for i, d in enumerate(self.scrambling_depths):
			self.log_this_depth(res[i], states[i], times[i], d)

		S_mu, S_conf = self.S_confidence(self.S_dist(res, self.scrambling_depths))
		self.log(f"S: {S_mu:.2f} p/m {S_conf:.2f}", with_timestamp=False)
		self.log.verbose(f"Evaluation runtime\n{self.tt}")

		return res, states, times

	def log_this_depth(self, res: np.ndarray, states: np.ndarray, times: np.ndarray, depth: int):
		"""Logs summary statistics for given deth

		:param res:  Vector of results
		:param states: Vector of seen states for each game
		:param times: Vector of runtimes for each game
		:param depth:  Scrambling depth at which results were generated
		"""
		share_completed = np.count_nonzero(res!=-1)*100/len(res)
		won_games = res[res!=-1]
		self.log(f"Scrambling depth {depth}", with_timestamp=False)
		self.log(f"\tShare completed: {share_completed:.2f} % {bernoulli_error(share_completed/100, len(res), 0.05, stringify=True)} (approx. 95 % CI)", with_timestamp=False)
		if won_games.size:
			mean_turns = won_games.mean()
			median_turns = np.median(won_games)
			std_turns = won_games.std()
			self.log(
				f"\tTurns to win: "\
				f"{mean_turns:.2f} +/- {std_turns:.1f} (std.), Median: {median_turns:.0f}"
				, with_timestamp=False
			)

		safe_times = times != 0
		states_per_sec = states[safe_times] / times[safe_times]
		self.log(
			f"\tStates seen: Pr. game: {states.mean():.2f} +/- {states.std():.0f} (std.), "\
			f"Pr. sec.: {states_per_sec.mean():.2f} +/- {states_per_sec.std():.0f} (std.)", with_timestamp=False)
		self.log(f"\tTime:  {times.mean():.2f} +/- {times.std():.2f} (std.)", with_timestamp=False)
	
	@staticmethod
	def S_dist(res: np.ndarray, scrambling_depths: np.ndarray) -> np.ndarray:
		"""
		Computes sum score game wise, that is it returns an array of length self.n_games
		It assumes that all srambling depths lower that self.scrambling_depths[0] are always solved
		and that all depths above self.scrambling_depths[-1] are never solved
		Overall S is the mean of the returned array
		:param res: Numpy array of evaluation results as returned by self.eval
		:param scrambling_depths: The scrambling depths used for the evaluation
		:return: Numpy array of length self.n_games
		"""
		solved = res != -1
		lower_depths = scrambling_depths[0] - 1
		return solved.sum(axis=0) + lower_depths

	@staticmethod
	def S_confidence(S_dist: np.ndarray, alpha=0.05):
		"""
		Calculates mean and confidence interval assuming normality on a distribution of S values as given by Evaluator.S_dist
		:param S_dist: numpy array of S values
		:param alpha: confidence interval
		:return: mean and z * sigma
		"""
		mu = np.mean(S_dist)
		std = np.std(S_dist)
		z = stats.norm.ppf(1 - alpha / 2)
		return mu, z * std / np.sqrt(len(S_dist))

	@classmethod
	def plot_evaluators(cls, eval_results: dict, eval_states: dict, eval_times: dict, eval_settings: dict, save_dir: str, title: str='') -> list:
		"""
		Plots evaluation results
		:param eval_results:   { agent name: [steps to solve, -1 for unfinished] }
		:param eval_states:    { agent name: [states seen during solving] }
		:param eval_times:     { agent name: [time spent solving] }
		:param eval_settings:  { agent name: { 'n_games': int, 'max_time': float, 'scrambling_depths': np.ndarray } }
		:param save_dir:       Directory in which to save plots
		:param title:          If given, overrides auto generated title in (depth, winrate) plot
		:return:               Locations of saved plots
		"""
		assert eval_results.keys() == eval_results.keys() == eval_times.keys() == eval_settings.keys(), "Keys of evaluation dictionaries should match"
		os.makedirs(save_dir, exist_ok=True)

		tab_colours = list(mcolour.TABLEAU_COLORS)
		colours = [tab_colours[i%len(tab_colours)] for i in range(len(eval_results))]

		save_paths = [
			cls._plot_depth_win(eval_results, save_dir, eval_settings, colours, title),
			cls._sol_length_boxplots(eval_results, save_dir, eval_settings, colours),
		]
		# Only plot (time, winrate) and S if shapes are the same
		if cls.check_equal_settings(eval_settings):
			save_paths.extend([
				cls._time_winrate_plot(eval_results, eval_times, save_dir, eval_settings, colours),
				cls._states_winrate_plot(eval_results, eval_states, save_dir, eval_settings, colours),
				cls._S_hist(eval_results, save_dir, eval_settings, colours),
			])
		
		return save_paths
	
	@classmethod
	def _plot_depth_win(cls, eval_results: dict, save_dir: str, eval_settings: dict, colours: list, title: str='') -> str:
		first_key = list(eval_results.keys())[0]  # Used to get the settings if the settings are the same
		# depth, win%-graph
		games_equal, times_equal = cls.check_equal_settings(eval_settings)
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		ax.set_ylabel(f"Percentage of {eval_settings[first_key]['n_games']} games won" if games_equal else "Percentage of games won")
		ax.set_xlabel(f"Scrambling depth: Number of random rotations applied to cubes")
		ax.locator_params(axis='x', integer=True, tight=True)

		for i, (agent, results) in enumerate(eval_results.items()):
			used_settings = eval_settings[agent]
			color = colours[i]
			win_percentages = (results != -1).mean(axis=1) * 100

			ax.plot(used_settings['scrambling_depths'], win_percentages, linestyle='dashdot', color=color)
			ax.scatter(used_settings['scrambling_depths'], win_percentages, color=color, label=agent)
		ax.legend()
		ax.set_ylim([-5, 105])
		ax.grid(True)
		ax.set_title(title if title else (f"Percentage of cubes solved in {eval_settings[first_key]['max_time']:.2f} seconds" if times_equal else "Cubes solved"))
		fig.tight_layout()

		path = os.path.join(save_dir, "eval_winrates.png")
		plt.savefig(path)
		plt.clf()

		return path

	@classmethod
	def _sol_length_boxplots(cls, eval_results: dict, save_dir: str, eval_settings: dict, colours: list) -> str:
		# Solution length boxplots
		plt.rcParams.update(rc_params_small)
		max_width = 2
		width = min(len(eval_results), max_width)
		height = (len(eval_results)+1) // width if width == max_width else 1
		positions = [(i, j) for i in range(height) for j in range(width)]
		fig, axes = plt.subplots(height, width, figsize=(width*10, height*6))

		max_sollength = 50
		agents, agent_results = list(zip(*eval_results.items()))
		agent_results = tuple(x.copy() for x in agent_results)
		for res in agent_results:
			res[res > max_sollength] = max_sollength
		ylim = np.array([-0.02, 1.02]) * max([res.max() for res in agent_results])
		min_ = min([x["scrambling_depths"][0] for x in eval_settings.values()])
		max_ = max([x["scrambling_depths"][-1] for x in eval_settings.values()])
		xticks = np.arange(min_, max_+1, max(np.ceil((max_-min_+1)/8).astype(int), 1))
		for used_settings, (i, position) in zip(eval_settings.values(), enumerate(positions)):
			# Make sure axes are stored in a matrix, so they are easire to work with, and select axes object
			if len(eval_results) == 1:
				axes = np.array([[axes]])
			elif len(eval_results) <= width and i == 0:
				axes = np.expand_dims(axes, 0)
			ax = axes[position]
			if position[1] == 0:
				ax.set_ylabel(f"Solution length")
			if position[0] == height - 1 or len(eval_results) <= width:
				ax.set_xlabel(f"Scrambling depth")
			ax.locator_params(axis="y", integer=True, tight=True)

			try:
				agent, results = agents[i], agent_results[i]
				assert type(agent) == str, str(type(agent))
				ax.set_title(agent if axes.size > 1 else "Solution lengths for " + agent)
				results = [depth[depth != -1] for depth in results]
				ax.boxplot(results)
				ax.grid(True)
			except IndexError:
				pass
			ax.set_ylim(ylim)
			ax.set_xlim([used_settings["scrambling_depths"].min()-1, used_settings["scrambling_depths"].max()+1])

		plt.setp(axes, xticks=xticks, xticklabels=[str(x) for x in xticks])
		plt.rcParams.update(rc_params)
		if axes.size > 1:
			fig.suptitle("Solution lengths")
		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		path = os.path.join(save_dir, "eval_sollengths.png")
		plt.savefig(path)
		plt.clf()

		return path

	@classmethod
	def _time_winrate_plot(cls, eval_results: dict, eval_times: dict, save_dir: str, eval_settings: dict, colours: list) -> str:
		# Make a (time spent, winrate) plot
		plt.figure(figsize=(19.2, 10.8))
		for (agent, res), times, colour in zip(eval_results.items(), eval_times.values(), colours):
			sort_idcs = np.argsort(times[-1])  # Have lowest usage times first
			wins, times = (res!=-1)[-1, sort_idcs], times[-1, sort_idcs]  # Delve too greedily and too deep into the cube
			cumulative_winrate = np.cumsum(wins) / len(wins) * 100
			plt.plot(times, cumulative_winrate, "o-", linewidth=3, color=colour, label=agent)
		plt.xlabel("Time used [s]")
		plt.ylabel("Winrate [%]")
		plt.ylim([-5, 105])
		plt.legend()
		plt.title("Winrate against time used for solving")
		plt.grid(True)
		plt.tight_layout()
		path = os.path.join(save_dir, "time_winrate.png")
		plt.savefig(path)
		plt.clf()
		
		return path

	@classmethod
	def _states_winrate_plot(cls, eval_results: dict, eval_states: dict, save_dir: str, eval_settings: dict, colours: list) -> str:
		# Make a (time spent, winrate) plot
		plt.figure(figsize=(19.2, 10.8))
		for (agent, res), states, colour in zip(eval_results.items(), eval_states.values(), colours):
			sort_idcs = np.argsort(states[-1])  # Have lowest usage times first
			wins, states = (res!=-1)[-1, sort_idcs], states[-1, sort_idcs]  # Delve too greedily and too deep into the cube
			cumulative_winrate = np.cumsum(wins) / len(wins) * 100
			plt.plot(states, cumulative_winrate, "o-", linewidth=3, color=colour, label=agent)
		plt.xlabel("States explored [s]")
		plt.ylabel("Winrate [%]")
		plt.ylim([-5, 105])
		plt.legend()
		plt.title("Winrate against states seen during solving")
		plt.grid(True)
		plt.tight_layout()
		path = os.path.join(save_dir, "states_winrate.png")
		plt.savefig(path)
		plt.clf()
		
		return path
		
	@classmethod
	def _S_hist(cls, eval_results: dict, save_dir: str, eval_settings: dict, colours: list) -> str:
		# Histograms of S
		games_equal, times_equal = cls.check_equal_settings(eval_settings)
		normal_pdf = lambda x, mu, sigma: np.exp(-1/2 * ((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		sss = np.array([Evaluator.S_dist(results, eval_settings[agent]['scrambling_depths']) for i, (agent, results) in enumerate(eval_results.items())])
		mus, confs = zip(*[Evaluator.S_confidence(ss) for ss in sss])
		stds = [ss.std() for ss in sss]
		lower, higher = sss.min() - 2, sss.max() + 2
		bins = np.arange(lower, higher+1)
		ax.hist(x           = sss.T,
				bins        = bins,
				density     = True,
				color       = colours,
				edgecolor   = "black",
				linewidth   = 2,
				align       = "left",
				label       = [f"{agent}: S = {mus[i]:.2f} p/m {confs[i]:.2f}" for i, agent in enumerate(eval_results.keys())])
		highest_y = 0
		for i in range(len(eval_results)):
			if stds[i] > 0:
				x = np.linspace(lower, higher, 500)
				y = normal_pdf(x, mus[i], stds[i])
				x = x[~np.isnan(y)]
				y = y[~np.isnan(y)]
				plt.plot(x, y, color="black", linewidth=9)
				plt.plot(x, y, color=colours[i], linewidth=5)
				highest_y = max(highest_y, y.max())
		ax.set_xlim([lower, higher])
		ax.set_xticks(bins)
		ax.set_title(f"Single game S distributions (evaluated on {eval_settings[list(eval_settings.keys())[0]]['max_time']:.2f} s per game)"
		             if times_equal else "Single game S distributions")
		ax.set_xlabel("S")
		ax.set_ylim([0, highest_y*(1+0.1*max(3, len(eval_results)))])  # To make room for labels
		ax.set_ylabel("Frequency")
		ax.legend(loc=2)
		path = os.path.join(save_dir, "eval_S.png")
		plt.savefig(path)
		plt.clf()

		return path

	@staticmethod
	def check_equal_settings(eval_settings: dict):
		# Super simple looper just to hide the ugliness
		games, times = list(), list()
		for setting in eval_settings.values():
			games.append(setting['max_time'])
			times.append(setting['n_games'])
		return games.count(games[0]) == len(games), times.count(times[0]) == len(times)
