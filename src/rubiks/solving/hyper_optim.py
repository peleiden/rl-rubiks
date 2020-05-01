import os
from glob import glob as glob # glob
from dataclasses import dataclass, field
from typing import Callable, List
import argparse
import json # For print

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.solving.agents import DeepAgent, Agent
from src.rubiks.utils.logger import Logger, NullLogger
from src.rubiks.solving.search import Searcher, MCTS
from src.rubiks.model import Model

class Optimizer:
	def __init__(self,
			# Maximizes target function
			target_function: Callable[[dict], float],
			parameters: dict, #str name : tuple limits

			logger: Logger=NullLogger(),
		):
		self.target_function = target_function
		self.parameters = parameters

		self.optimal = None
		self.highscore = None

		# For evaluation use
		self.evaluator = None
		self.searcher_class = None
		self.persistent_searcher_params = None
		self.agent_class = None

		self.score_history = list()
		self.parameter_history = list()

		self.logger=logger
		self.logger.log(f"Optimizer {self} created parameters: {self._format_params(self.parameters)}")

	def optimize(self, iterations: int):
		raise NotImplementedError("To be implemented in child class")

	def objective_from_evaluator(self, evaluator: Evaluator, searcher_class, persistent_searcher_params: dict, agent_class = DeepAgent):
		self.evaluator = evaluator
		self.searcher_class = searcher_class
		self.agent_class = agent_class
		self.persistent_searcher_params = persistent_searcher_params

		def target_function(self, searcher_params):
			searcher = self.seacher_class(**self.persistent_searcher_params, **seacher_params)
			agent = self.agent_class(searcher)
			res = self.evaluator.eval(agent)
			return (res != -1).mean()

		self.target_function = target_function

	def plot_optimization(self):
		raise NotImplementedError

	@staticmethod
	def _format_params(params: str):
		return json.dumps(params, indent=4, sort_keys=True)

class BayesianOptimizer(Optimizer):
	""" An optimizer using https://github.com/fmfn/BayesianOptimization."""
	def __init__(self,
			# Maximizes target function
			target_function: Callable[[dict], float],
			parameters: dict,

			alpha=1e-6,
			n_restarts = 20,
			acquisition: str='ei',

			logger: Logger=NullLogger(),
			verbose: bool=True,

		):
		"""Set op BO class, set up utility function (acqusition function) and gaussian process.

		:param float alpha:  Handles how much noise the GP can deal with
		:param int n_restarts: Higher => more expensive, but more accurate
		"""
		super().__init__(target_function, parameters, logger)

		self.optimizer = BayesianOptimization(
			f=None,
			pbounds=dict,
			verbose=0,
		)
		self.optimizer.set_gp_params(alpha=alpha, n_restarts=n_restarts)
		self.utility = UtilityFunction(kind=acquisition, kappa=2.5, xi=0)

		self.logger(f"Created Bayesian Optimizer with alpha={alpha} and {n_restarts} for each optimization. Acquisition function is {acquisition}.")

	def optimize(self, iterations: int):
		for i in self.iterations:
			next_params = optimizer.suggest(self.utility)
			self.parameter_history.append(next_params)
			self.logger("Optimization {i}: Chosen parameters:\t: {self._format_params(next_params)}")

			score = self.target_function(**next_params)
			self.score_history.append(score)
			self.logger("Optimization {i}: Score: {score}")

			self.optimizer.register(params=next_params, target=score)

		high_idx = np.argmax(self.scores)
		self.highscore = self.scores[high_idx]
		self.optimal = self.parameter_history[high_idx]

		self.logger("Optimization done. Best parameters: {self.optimal} with score {self.highscore}")

		return self.optimal
	def __str__(self):
		return f"BayesianOptimizer()"

def MCTS_optimize():
	#Lot of overhead just default argument niceness: latest model is latest
	from src.rubiks.runeval import train_folders
	model_path = ''
	if train_folders:
		for folder in [train_folders[-1]] + glob(f"{train_folders[-1]}/*/"):
				if os.path.isfile(os.path.join(folder, 'model.pt')):
					model_path  = os.path.join(folder)
					break
	parser = argparse.ArgumentParser(description='Optimize Monte Carlo Tree Search for one model')
	parser.add_argument('--location', help='Location for model.pt. Results will also be saved here',
		type=str, default=model_path)
	parser.add_argument('--iterations', help='Number of iterations of Bayesian Optimization',
		type=int, default=25)
	args = parser.parse_args()

	params = {
		'c': (0.2, 0.8),
		'nu': (0, 0.01),
	}
	persistent_params = {
		'net' : Model.load(args.location),
		'complete_graph': False,
		'search_graph': False,
	}
	logger = Logger(os.path.join(args.location, 'optimizer.log'), 'Optimization')

	evaluator = Evaluator(n_games=20, max_time=1, scrambling_depths=range(12, 20))
	optimizer = BayesianOptimizer(target_function=None, parameters=params, logger=logger)
	optimizer.objective_from_evaluator(evaluator, MCTS, persistent_params, persistent_searcher_params=persistent_params)
	optimizer.optimize(parser.iterations)

if __name__ == '__main__':
	MCTS_optimize()


