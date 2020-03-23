import pytest
import numpy as np

from src.rubiks.cube.cube import Cube
from src.rubiks.post_train.agents import Agent, RandomAgent


class TestAgent:
	
	def test_actions(self):
		a = Agent()
		state = Cube.get_solved()
		# Make sure every aciton in action space is possible
		for action in a.action_space:
			state = Cube.rotate(state, *action)
		assert a.action_dim == 12
		
	def test_act(self):
		a = Agent()
		with pytest.raises(NotImplementedError) as e_info:
			a.act(None)

class TestRandomAgent:
	
	def test_init(self):
		a = RandomAgent()
		assert isinstance(a, Agent)
	def test_act(self):
		np.random.seed(42)
		a = RandomAgent()
		state = Cube.get_solved()
		for _ in range(10):
			state = Cube.rotate(state, *a.act(state))
