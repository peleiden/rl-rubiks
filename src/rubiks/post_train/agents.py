import numpy as np

from src.rubiks.cube.cube import Cube



class Agent:
	action_space, action_dim = Cube.action_space, Cube.action_dim

	def __init__(self, model_needed: bool = True):
		self.model_needed = model_needed
		self.model_env = Cube if model_needed else None

	def act(self, state: np.array):
		raise NotImplementedError

	def __str__(self):
		return f"{self.__class__.__name__}(Agent)"

class RandomAgent(Agent):
	def __init__(self, **kwargs):
		super().__init__(model_needed=False, **kwargs)
		
	def act(self, state: np.array):
		return self.action_space[np.random.randint(self.action_dim)]

class SimpleBFS(Agent):
	def __init__(self, **kwargs):
		super().__init__(model_needed=True, **kwargs)

	def act(self, state: np.array):
		return NotImplementedError


class DeepCube(Agent):
	def __init__(self, net = None, **kwargs):
		super().__init__(model_needed=False, **kwargs)

	def act(self, state: np.array):
		return NotImplementedError

	def update_net(self, net):
		raise NotImplementedError