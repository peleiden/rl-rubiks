import os
from glob import glob as glob # glob
from dataclasses import dataclass, field
from typing import Callable, List
import argparse
import json # For print

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

from librubiks.utils import Logger, NullLogger

from librubiks.solving.evaluation import Evaluator
from librubiks.solving.agents import DeepAgent

import librubiks.solving.search as search
from librubiks.model import Model

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

	def objective_from_evaluator(self, evaluator: Evaluator, searcher_class, persistent_searcher_params: dict, param_prepper: Callable=lambda x: None,  agent_class = DeepAgent):
		self.evaluator = evaluator
		self.searcher_class = searcher_class
		self.agent_class = agent_class
		self.persistent_searcher_params = persistent_searcher_params

		def target_function(searcher_params):
			param_prepper(searcher_params)
			searcher = self.searcher_class(**self.persistent_searcher_params, **searcher_params)
			agent = self.agent_class(searcher)
			res = self.evaluator.eval(agent)
			won = res != -1
			return won.mean() if won.any() else 0

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

			alpha: float =1e-6,
			n_restarts: int = 20,
			acquisition: str='ei',

			logger: Logger=NullLogger(),

		):
		"""Set op BO class, set up utility function (acqusition function) and gaussian process.

		:param float alpha:  Handles how much noise the GP can deal with
		:param int n_restarts: Higher => more expensive, but more accurate
		"""
		super().__init__(target_function, parameters, logger)

		self.optimizer = BayesianOptimization(
			f=None,
			pbounds=parameters,
			verbose=0,
		)
		self.optimizer.set_gp_params(alpha=alpha, n_restarts_optimizer=n_restarts)
		self.utility = UtilityFunction(kind=acquisition, kappa=2.5, xi=0)

		self.logger(f"Created Bayesian Optimizer with alpha={alpha} and {n_restarts} restarts for each optimization. Acquisition function is {acquisition}.")

	def optimize(self, iterations: int):
		for i in range(iterations):
			next_params = self.optimizer.suggest(self.utility)
			self.parameter_history.append(next_params)
			self.logger(f"Optimization {i}: Chosen parameters:\t: {self._format_params(next_params)}")

			score = self.target_function(next_params)
			self.score_history.append(score)
			self.logger(f"Optimization {i}: Score: {score}")

			self.optimizer.register(params=next_params, target=score)

		high_idx = np.argmax(self.score_history)
		self.highscore = self.score_history[high_idx]
		self.optimal = self.parameter_history[high_idx]

		self.logger(f"Optimization done. Best parameters: {self._format_params(self.optimal)} with score {self.highscore}")

		return self.optimal
	def __str__(self):
		return f"BayesianOptimizer()"

def searcher_optimize():
	#Lot of overhead just for default argument niceness: latest model is latest
	from runeval import train_folders

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
	parser.add_argument('--searcher', help='Name of searcher for agent corresponding to searcher class in librubiks.solving.search',
		type=str, default='AStar', choices = ['MCTS', 'AStar'])
	parser.add_argument('--depth', help='Single number corresponding to the depth at which to test',
		type=int, default=50)
	parser.add_argument('--eval_games', help='Number of games to evaluate at depth 50',
			type = int, default='20')
	args = parser.parse_args()

	searcher = args.searcher
	if searcher == 'MCTS':
		params = {
			'c': (0.1, 1),
		}
		def prepper(params): pass

		persistent_params = {
			'net' : Model.load(args.location),
			'search_graph': False,
		}
	elif searcher == 'AStar':
		params = {
			'lambda_': (0.1, 1),
			'expansions': (1, 250),
		}
		def prepper(params): params['expansions'] = int(params['expansions'])

		persistent_params = {
			'net' : Model.load(args.location),
		}
	else: raise NameError(f"{searcher} does not correspond to a known searcher, please pick either AStar og MCTS")

	logger = Logger(os.path.join(args.location, 'optimizer.log'), 'Optimization')
	logger.log(f"{searcher} optimization. Using network from {model_path}.")

	searcher = getattr(search, searcher)

	evaluator = Evaluator(n_games=args.eval_games, max_time=1, scrambling_depths=[args.depth])
	optimizer = BayesianOptimizer(target_function=None, parameters=params, logger=logger)
	optimizer.objective_from_evaluator(evaluator, searcher, persistent_params, param_prepper=prepper)
	optimizer.optimize(args.iterations)

if __name__ == '__main__':
	searcher_optimize()
