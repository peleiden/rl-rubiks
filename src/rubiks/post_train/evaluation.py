from os import cpu_count
import torch.multiprocessing as mp
import numpy as np

from src.rubiks.utils.logger import NullLogger, Logger
from src.rubiks.utils.ticktock import TickTock

from src.rubiks.cube.cube import Cube
from src.rubiks.post_train.agents import Agent, DeepCube

# Multiprocessing is silly, so all functions have to be top-level
# This also means all info has to be parsed in with a single argument
# https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
def _eval_game(cfg: (Agent, int, int)):
	agent, max_moves, depth = cfg
	turns_to_complete = 0  # 0 for unfinished
	state, _, _ = Cube.scramble(depth)
	for i in range(max_moves):
		action = agent.act(state)
		state = Cube.rotate(state, *action)
		if Cube.is_solved(state):
			turns_to_complete = i + 1
			break
	return turns_to_complete


class Evaluator:
	def __init__(self,
				 n_games					= 420,  # Nice
				 max_moves					= 200,
				 scrambling_depths			= range(1, 10),
				 verbose	 				= True,
				 logger: Logger 			= NullLogger()
		):

		self.n_games = n_games
		self.max_moves = max_moves

		self.tt = TickTock()
		self.log = logger
		self.verbose = verbose
		self.scrambling_depths = np.array(scrambling_depths)

		self.log("\n".join([
			"Creating evaluator",
			f"Games per scrambling depth: {self.n_games}",
			f"Scrambling depths: {scrambling_depths}",
			f"Max moves: {self.max_moves}",
		]))
	
	def eval(self, agent: Agent, max_threads: int = None):
		"""
		Evaluates an agent
		"""
		self.log(f"Evaluating {self.n_games*len(self.scrambling_depths)} games with agent {agent}")
		
		# Builds configurations for runs
		cfgs = []
		for i, d in enumerate(self.scrambling_depths):
			for _ in range(self.n_games):
				cfgs.append((agent, self.max_moves, d))
		
		self.tt.section(f"Evaluation of {agent}")
		if agent.with_mt:
			with mp.Pool(max_threads if max_threads and max_threads < cpu_count() else cpu_count()) as p:
				res = p.map(_eval_game, cfgs)
		else:
			res = [_eval_game(cfg) for cfg in cfgs]
		self.tt.end_section(f"Evaluation of {agent}")
		res = np.reshape(res, (len(self.scrambling_depths), self.n_games))
		
		self.log(f"Evaluation results")
		for i, d in enumerate(self.scrambling_depths):
			self.log(f"Scrambling depth {d}", with_timestamp=False)
			self.log(f"\tShare completed: {np.count_nonzero(res[i]!=0)*100/len(res[i]):.2f} %", with_timestamp=False)
			self.log(f"\tMean turns to complete (ex. unfinished): {res[i][res[i]!=0].mean():.2f}", with_timestamp=False)
			self.log(f"\tMedian turns to complete (ex. unfinished): {np.median(res[i][res[i]!=0]):.2f}", with_timestamp=False)
		if self.verbose:
			self.log(f"Evaluation runtime\n{self.tt}")
		
		return res
	
	def eval_hists(self, eval_results: dict):
		"""
		{agent: results from self.eval}
		"""
		raise NotImplementedError
	
	def _get_eval_fn(self, agent: Agent):
		def eval_fn(depth: int):
			turns_to_complete = 0  # 0 for unfinished
			state = Cube.scramble(depth)
			for i in range(self.max_moves):
				action = agent.act(state)
				state = Cube.rotate(state, *action)
				if Cube.is_solved(state):
					turns_to_complete = i + 1
					break
			return turns_to_complete
		return eval_fn
			

if __name__ == "__main__":
	mp.set_start_method("spawn")  # Necessary for cuda to work in mp mode
	from src.rubiks.post_train.agents import RandomAgent
	e = Evaluator(n_games = 1000,
				  max_moves=10,
				  logger = Logger("local_evaluation/randomagent.log", "Testing the RandomAgent"),
				  scrambling_depths = range(1, 10)
	)
	# results = e.eval(RandomAgent(), 6)
	results = e.eval(DeepCube.from_saved("local_train"))
	# TODO: Boxplot over completion turns for each scrambling depth
