import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass

from src.rubiks.cube.cube import Cube
from src.rubiks.utils.ticktock import TickTock

class Node:
	state: np.ndarray
	N: np.ndarray
	W: np.ndarray
	L: np.ndarray
	P: np.ndarray
	def __init__(self, state, value: float, policy: np.ndarray, from_node=None, action_idx: int=None):
		self.state = state
		self.value = value
		self.P = policy
		# self.neighs[i] is a tuple containing the state obtained by the action Cube.action_space[i]
		# Tuples are used, so they can be used for lookups
		self.neighs = [None] * 12
		self.N = np.ones(12)
		self.W = np.empty(12)
		self.L = np.zeros(12)
		if action_idx is not None:
			from_action_idx = action_idx + 1 if action_idx % 2 == 0 else action_idx - 1
			self.neighs[from_action_idx] = tuple(from_node.state)
			self.W[from_action_idx] = from_node.value





class Searcher:
	def __init__(self, agent_class):
		self.action_queue = deque()
		self.agent_class = agent_class

	def search(self, state: np.ndarray, time_limit: int):
		raise NotImplementedError

class RandomDFS(Searcher):
	def search(self, state: np.ndarray, time_limit: int):
		tt = TickTock()
		tt.tick()
		while tt.tock() < time_limit:
			action = np.random.randint(Cube.action_dim)
			state = Cube.rotate(state, *Cube.action_space[action])

			self.action_queue.append(action)
			if Cube.is_solved(state): break

class BFS(Searcher):
	def search(self, state: np.ndarray, time_limit: int):
		raise NotImplementedError

class MCTS(Searcher):
	def __init__(self, *args, **kwargs):
		self.states = dict()
		super().__init__(*args, **kwargs)


	def search(self, state: np.ndarray, time_limit: int):
		tt = TickTock()
		tt.tick()
		self.states = Node(state, )
		while tt.tock() < time_limit:
			pass
