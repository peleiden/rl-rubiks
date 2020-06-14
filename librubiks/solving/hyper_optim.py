import os
from glob import glob as glob # glob
from typing import Callable, List
from ast import literal_eval
from copy import copy
import argparse
import json # For print

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

from librubiks.utils import Logger, NullLogger, set_seeds

from librubiks.solving.evaluation import Evaluator

from librubiks.solving import agents
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
		self.persistent_agent_params = None
		self.agent_class = None
		self.param_prepper = None

		self.score_history = list()
		self.parameter_history = list()

		self.logger=logger
		self.logger.log(f"Optimizer {self} created parameters: {self.format_params(self.parameters)}")

	def optimize(self, iterations: int):
		raise NotImplementedError("To be implemented in child class")

	def objective_from_evaluator(self, evaluator: Evaluator, agent_class, persistent_agent_params: dict, param_prepper: Callable=lambda x: x, optim_lengths: bool=False):
		self.evaluator = evaluator
		self.agent_class = agent_class
		self.persistent_agent_params = persistent_agent_params
		self.param_prepper = param_prepper

		def target_function(agent_params):
			agent = self.agent_class(**self.persistent_agent_params, **self.param_prepper(copy(agent_params)))
			res, _, _ = self.evaluator.eval(agent)
			res = res.ravel()
			won = res != -1
			solve = won.mean() if won.any() else 0
			meanlength = res[won].mean() if solve else -1
			self.logger.log(f"\t RESULTS: Solved:: {solve}, Mean length {meanlength}")
			if optim_lengths: return solve/meanlength
			return solve

		self.target_function = target_function

	def plot_optimization(self):
		raise NotImplementedError

	@staticmethod
	def format_params(params: str, prep=None):
		if prep is not None: params = prep(copy(params))
		return json.dumps(params, indent=4, sort_keys=True)

class BayesianOptimizer(Optimizer):
	""" An optimizer using https://github.com/fmfn/BayesianOptimization."""
	def __init__(self,
			# Maximizes target function
			target_function: Callable[[dict], float],
			parameters: dict,

			alpha: float =1e-3,
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
		self.utility = UtilityFunction(kind=acquisition, kappa=2.5, xi=0.05)

		self.logger(f"Created Bayesian Optimizer with alpha={alpha} and {n_restarts} restarts for each optimization. Acquisition function is {acquisition}.")

	def optimize(self, iterations: int):
		for i in range(iterations):
			next_params = self.optimizer.suggest(self.utility)
			self.parameter_history.append(next_params)
			self.logger(f"Optimization {i}: Chosen parameters:\t: {self.format_params(next_params, prep=self.param_prepper)}")

			score = self.target_function(next_params)
			self.score_history.append(score)
			self.logger(f"Optimization {i}: Score: {score}")

			self.optimizer.register(params=next_params, target=score)

		high_idx = np.argmax(self.score_history)
		self.highscore = self.score_history[high_idx]
		self.optimal = self.parameter_history[high_idx]

		self.logger(f"Optimization done. Best parameters: {self.format_params(self.optimal, prep=self.param_prepper)} with score {self.highscore}")

		return self.optimal

	def __str__(self):
		return f"BayesianOptimizer()"

def agent_optimize():
	"""
	Main way to run optimization. Hard coded to run optimization at 1 sec per game, but other behaviour can be set with CLI arguments seen by
	running `python librubiks/solving/hyper_optim.py --help`.
	Does not support config arguments.
	NB: The path here is different to the one in runeval and runtrain:
	It needs to be to folder containing model.pt! It doesen't work with parent folder.

	Can work with runeval through
	```
	python librubiks/solving/hyper_optim.py --location example/net1/
	python runeval.py --location example/ --optimized_params True
	```
	"""
	set_seeds()

	#Lot of overhead just for default argument niceness: latest model is latest
	from runeval import train_folders

	model_path = ''
	if train_folders:
		for folder in [train_folders[-1]] + glob(f"{train_folders[-1]}/*/"):
			if os.path.isfile(os.path.join(folder, 'model.pt')):
				model_path = os.path.join(folder)
				break

	parser = argparse.ArgumentParser(description='Optimize Monte Carlo Tree Search for one model')
	parser.add_argument('--location', help='Folder which includes  model.pt. Results will also be saved here',
		type=str, default=model_path)
	parser.add_argument('--iterations', help='Number of iterations of Bayesian Optimization',
		type=int, default=100)
	parser.add_argument('--agent', help='Name of agent corresponding to agent class in librubiks.solving.agents',
		type=str, default='AStar', choices = ['AStar', 'MCTS', 'EGVM'])
	parser.add_argument('--depth', help='Single number corresponding to the depth at which to test. If 0: run this at deep',
		type=int, default=0)
	parser.add_argument('--eval_games', help='Number of games to evaluate at depth',
			type = int, default='100')
	parser.add_argument('--save_optimal', help='If Tue, saves a JSON of optimal hyperparameters usable for runeval',
			type=literal_eval, default=True, choices = [True, False])
	parser.add_argument('--use_best', help="Set to True to use model-best.pt instead of model.pt.", type=literal_eval, default=True,
			choices = [True, False])
	parser.add_argument('--optim_lengths', help="Set to true to optimize against sol percentage / solution length. Else, simply use sol %", type=literal_eval,
			default=True, choices = [True, False])

	args = parser.parse_args()

	agent_name = args.agent
	if agent_name == 'MCTS':
		params = {
			'c': (0.1, 4),
		}
		def prepper(params): return params

		persistent_params = {
			'net' : Model.load(args.location, load_best=args.use_best),
			'search_graph': False,
		}
	elif agent_name == 'AStar':
		params = {
			'lambda_': (0, 0.5),
			'expansions': (0, 4),
		}
		def prepper(params):
			params['expansions'] = int(10**params['expansions'])
			return params

		persistent_params = {
			'net' : Model.load(args.location, load_best=args.use_best),
		}
	elif agent_name == 'EGVM':
		params = {
				'epsilon': (0, 1),
				'workers': (1, 2000),
				'depth': (1, 1000),
			}

		def prepper(params):
			params['workers'] = int(params['workers'])
			params['depth'] = int(params['depth'])
			return params

		persistent_params = {
			'net' : Model.load(args.location, load_best=args.use_best),
		}
	else:
		raise NameError(f"{agent_name} does not correspond to a known agent, please pick either AStar, MCTS or EGVM")

	logger = Logger(os.path.join(args.location, f'{agent_name}_optimization.log'), 'Optimization')

	logger.log(f"{agent_name} optimization. Using network from {model_path}.")
	logger.log(f"Received arguments: {vars(args)}")

	agent = getattr(agents, agent_name)

	evaluator = Evaluator(n_games=args.eval_games, max_time=5, scrambling_depths=range(0) if args.depth == 0 else [args.depth])
	optimizer = BayesianOptimizer(target_function=None, parameters=params, logger=logger)
	optimizer.objective_from_evaluator(evaluator, agent, persistent_params, param_prepper=prepper, optim_lengths=args.optim_lengths)
	optimizer.optimize(args.iterations)

	if args.save_optimal:
		with open(os.path.join(args.location, f'{agent_name}_params.json'), 'w') as outfile:
			json.dump(prepper(copy(optimizer.optimal)), outfile)

if __name__ == '__main__':
	agent_optimize()
