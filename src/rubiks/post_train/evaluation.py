import numpy as np
import multiprocessing as mp

from os import cpu_count, makedirs

from src.rubiks.utils.logger import NullLogger, Logger
from src.rubiks.utils.ticktock import TickTock

from src.rubiks.cube import RubiksCube
from src.rubiks.post_train.agents import Agent


class Evaluator:
	def __init__(self, 
			agent: Agent,
			max_moves: int				= 200,
			scrambling_procedure: dict 	= None,
			verbose: bool 				= True,
			logger: Logger 				= NullLogger()
		):

		self.agent = agent
		self.max_moves = max_moves if max_moves is not None else 200

		self.tt = TickTock()
		self.log = logger
		self.verbose = verbose
		if scrambling_procedure is not None:
			RubiksCube.scrambling_procedure = scrambling_procedure

		self.log(f"Creating evaluator: Scrambling procedure: {RubiksCube.scrambling_procedure}, max_moves: {self.max_moves}, agent: {self.agent} ")

	
	def eval(self, N_games: int, n_threads: int = cpu_count()):
		if self.verbose: self.log(f"Evaluation: Beginning {N_games} games on {n_threads} threads")

		self.tt.tick()
		with mp.Pool(n_threads) as p:
			results = p.map(self._run_N_games, [int(N_games) // n_threads]*n_threads) #Divide games equally between threads
		time = self.tt.tock()
		
		if self.verbose: self.log(f"Evaluation completed in {time:.4} s")

		all_results = np.concatenate(results)
		
		self.log(f"Evaluation results out of {len(all_results)} games:\n\
		\tnumber of wins: {len(all_results[all_results != 0])}\n\
		\tmoves to win-distribution [[moves], [counts]]:{[ list(res) for res in np.unique(all_results[all_results != 0], return_counts = True) ]} ")
		return all_results

	
	def _run_N_games(self, N: int):
		results = np.zeros(N) #0 represents not completed 
		cube = RubiksCube()

		for i in range(N):
			cube.reset()
			for j in range(self.max_moves):
				action = self.agent.act( cube.state )
				if cube.move(*action): 
					results[i] = j #saving number of steps required to win
					break
		return results

if __name__ == "__main__":
	from src.rubiks.post_train.agents import RandomAgent
	e = Evaluator(RandomAgent(),
		logger = Logger("local_evaluation/randomagent.log", "Testing the RandomAgent"),
		scrambling_procedure = {'N_scrambles':	(1, 3)} 
	) 
	results = e.eval(1e3)
