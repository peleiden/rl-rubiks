from dataclasses import dataclass, field
from typing import Callable, Iterable, List

import numpy as np

from src.rubiks.solving.utils.logger import Logger, NullLogger


@dataclass
class Parameter:
	"""
	A parameter which varied in the optimization
	"""
	name: str
	domain: Iterable

	value =  field(init=False) #Can be arbitrary type

class Optimizer:
	def __init__(self,
			# Maximizes target function
			target_function: Callable[[dict], float],
			parameters: List[Parameter],

			logger: Logger=NullLogger(),
			with_mt: bool=False,
		):
		self.target_function = target_function
		self.with_mt = with_mt
		self.parameters = parameters
		self.logger=logger

		self.score_history = list()
		self.parameter_history = list() # Note: Length of this will be a multiple of score_history if run with calls_per_iter > 1.

		self.logger.log(f"Optimizer {self} created with parameters:\n{'\n'.join(self.parameters)}\n")

	def optimize(self, iterations: int, calls_per_iter: int):
		raise NotImplementedError("Implement in child class")

	def plot_optimization(self):
		raise NotImplementedError


class GridSearch(Optimizer):

	def optimize(self, iterations: int, calls_per_iter: int):
		# Set up grid
		N_calls = calls_per_iter*iterations
		N_parameter_divisions = np.ones(len(parameters)*(N_calls // len(parameters))
		# The first parameters are prioritized when there is a rest
		N_parameter_divisions[:N_calls % len(parameters)] += 1
		# Create history before optimizing as optimization is independent of results
		self.parameter_history = [[] for _ in range(N_calls)]
		for i, parameter in enumerate(parameters):
			discrepancy = N_calls - len(parameter.domain)
			if discrepancy > 0:
				#TODO: Add the discrepancy to the next parameters
				pass
			#TODO: update parameter history

		#TODO: Run through parameter history

	def __str__(self):
		return "Gridsearch(Optimizer)"

class BayesianOptimizer(Optimizer):
	# def __init__(self,
			# target_function: Callable[[dict], float],
			# parameters: List[Parameter],
			# with_mt: bool=False
		# ):
		# super().__init__(target_function, parameters, with_mt)

	def optimize(self, iterations: int, calls_per_iter: int):
		raise NotImplementedError


