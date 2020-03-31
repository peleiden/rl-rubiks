import numpy as np
from collections import deque, defaultdict

from src.rubiks.cube.cube import Cube
from src.rubiks.utils.ticktock import TickTock
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
		super().__init__(*args, **kwargs)
		self.state_count = defaultdict(lambda: np.zeros(Cube.action_dim))
		self.state_maxval = defaultdict(lambda: np.zeros(Cube.action_dim))
		self.state_policy = dict()


	def search(self, state: np.ndarray, time_limit: int):
		tt = TickTock()
		tt.tick()
		while tt.tock() < time_limit:
			pass
