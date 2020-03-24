from abc import ABC

import numpy as np
import torch

from src.rubiks.cube.cube import Cube
from src.rubiks.model import Model
from src.rubiks.utils.device import cpu, gpu


class Agent:
	action_space, action_dim = Cube.action_space, Cube.action_dim
	# NN based agents see very little gain but much higher compute usage with standard mt implementation
	# TODO: Either stick to ST for these agents or find better solution
	with_mt = False

	def act(self, state: np.ndarray):
		raise NotImplementedError

	def __str__(self):
		return f"{self.__class__.__name__}"

class RandomAgent(Agent):
	with_mt = True
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		
	def act(self, state: np.ndarray):
		return self.action_space[np.random.randint(self.action_dim)]

class SimpleBFS(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def act(self, state: np.ndarray):
		return NotImplementedError

class DeepAgent(Agent):
	def __init__(self, net: Model, **kwargs):
		super().__init__(**kwargs)
		self.net = net

	def act(self, state: np.ndarray) -> (int, bool):
		raise NotImplementedError

	def update_net(self, net):
		raise NotImplementedError
	
	@classmethod
	def from_saved(cls, loc: str, **kwargs):
		net = Model.load(loc)
		net.to(gpu)
		return cls(net, **kwargs)



class PolicyCube(DeepAgent):
	# Pure neural net agent
	def __init__(self, net, sample_policy=False, **kwargs):
		super().__init__(net, **kwargs)
		self.sample_policy = sample_policy

	def act(self, state: np.ndarray) -> (int, bool):
		# child_states = np.array([Cube.rotate(state, *action) for action in Cube.action_space])
		# oh = Cube.as_oh(child_states).to(gpu)
		oh = Cube.as_oh(state).to(gpu)
		self.net.eval()
		with torch.no_grad():
			policy = self.net(oh, True, False)
			# vals = self.net(oh, False, True).squeeze()
			# print(vals)
			# policy = torch.nn.functional.softmax(self.net(oh, False, True).squeeze(), dim=0)
		if self.sample_policy:
			action = np.random.choice(12, p=policy.cpu().numpy())
		else:
			action = int(torch.argmax(policy))
		return Cube.action_space[action]


class DeepCube(DeepAgent):
	def __init__(self, net, sample_policy=False, **kwargs):
		super().__init__(net, **kwargs)
		self.sample_policy = sample_policy

	def act(self, state: np.ndarray) -> (int, bool):
		raise NotImplementedError

	


