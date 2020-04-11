from os import cpu_count
import torch.multiprocessing as mp
import numpy as np

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
	state, _, _ = Cube.scramble(depth)
	# breakpoint()
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
		self.log(f"{self.n_games*len(self.scrambling_depths)} games with max time per game {self.max_time}\nExpected time <~ {self.max_time*self.n_games*len(self.scrambling_depths)/60:.2f} min")

		# Builds configurations for runs
		cfgs = []
		# TODO: Pass a logger along to log progress
		for i, d in enumerate(self.scrambling_depths):
			for _ in range(self.n_games):
				cfgs.append((agent, self.max_time, d))

		max_threads = max_threads or cpu_count()
		threads = min(max_threads, cpu_count()) if agent.allow_mt() else 1
		self.tt.section(f"Evaluating {agent} on {threads} threads")
		if agent.allow_mt():
			self.log(f"Evaluating {agent} on {threads} threads")
			with mp.Pool(threads) as p:
				res = p.map(_eval_game, cfgs)
		else:
			res = []
			for i, cfg in enumerate(cfgs):
				self.tt.section(f"Evaluation of {agent}. Depth {cfg[2]}")
				self.log(f"Performing evaluation {i+1} / {len(cfgs)}. Depth: {cfg[2]}")
				res.append(_eval_game(cfg))
				self.tt.end_section(f"Evaluation of {agent}. Depth {cfg[2]}")
		res = np.reshape(res, (len(self.scrambling_depths), self.n_games))
		self.tt.end_section(f"Evaluating {agent} on {threads} threads")

		self.log(f"Evaluation results")
		for i, d in enumerate(self.scrambling_depths):
			self.log(f"Scrambling depth {d}", with_timestamp=False)
			self.log(f"\tShare completed: {np.count_nonzero(res[i]!=-1)*100/len(res[i]):.2f} %", with_timestamp=False)
			if res.any():
				self.log(f"\tMean turns to complete (ex. unfinished): {res[i][res[i]!=-1].mean():.2f}", with_timestamp=False)
				self.log(f"\tMedian turns to complete (ex. unfinished): {np.median(res[i][res[i]!=-1]):.2f}", with_timestamp=False)
		self.log.verbose(f"Evaluation runtime\n{self.tt}")

		return res

	def eval_hists(self, eval_results: dict):
		"""
		{agent: results from self.eval}
		"""
		raise NotImplementedError


if __name__ == "__main__":
	from src.rubiks.solving.agents import Agent, DeepAgent
	e = Evaluator(n_games = 5,
				  max_time = 1,
				  logger = Logger("local_evaluation/evaluations.log", "Testing MCTS", True),
				  scrambling_depths = range(1, 5)
	)
	# agent = PolicyCube.from_saved("local_train")
	# results = e.eval(agent, 1)
	agents = [
		Agent(RandomDFS()),
		Agent(BFS()),
		DeepAgent(PolicySearch.from_saved("local_train", False)),
		DeepAgent(PolicySearch.from_saved("local_train", True)),
		DeepAgent(MCTS.from_saved("local_train"))
	]
	for agent in agents:
		e.eval(agent)
	# results = e.eval(PolicyCube.from_saved("local_train"))
	# TODO: Boxplot with completion turns for each scrambling depth


