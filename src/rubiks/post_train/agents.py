import numpy as np

from src.rubiks.cube.cube import Cube



class Agent:
	action_space, action_dim = Cube.action_space, Cube.action_dim

	def act(self, state: np.ndarray):
		raise NotImplementedError

	def __str__(self):
		return f"{self.__class__.__name__} (Agent)"

class RandomAgent(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		
	def act(self, state: np.ndarray):
		return self.action_space[np.random.randint(self.action_dim)]

class SimpleBFS(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def act(self, state: np.ndarray):
		return NotImplementedError


class DeepCube(Agent):
	def __init__(self, net = None, **kwargs):
		super().__init__(**kwargs)

	def act(self, state: np.ndarray):
		return NotImplementedError

	def update_net(self, net):
		raise NotImplementedError
	
	@staticmethod
	def from_saved(loc: str):
		raise NotImplementedError


