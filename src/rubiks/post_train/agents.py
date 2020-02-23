import numpy as np

from src.rubiks.cube import RubiksCube



class Agent:
	action_space = list()
	for i in range(6): action_space.extend( [(i, True), (i, False)] )
	action_dim = len(action_space)

	def __init__(self, model_based: bool = True):
		self.model_based = model_based
		self.model_env = RubiksCube if model_based else None

	def act(self, state: iter):
		raise NotImplementedError
	
class RandomAgent(Agent):
	def __init__(self, **kwargs):
		super().__init__(model_based=False, **kwargs)
		
	def act(self, state: iter):
		return self.action_space[np.random.randint(self.action_dim)]

