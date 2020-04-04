import pytest
import numpy as np

from src.rubiks.cube.cube import Cube
from src.rubiks.solving.agents import Agent, RandomAgent, PolicyCube


class TestAgent:

	def test_actions(self):
		a = Agent()

	def test_act(self):
		a = Agent()
		with pytest.raises(NotImplementedError) as e_info:
			a.act(None)

class TestRandomAgent:

	def test_init(self):
		a = RandomAgent(2)
		assert isinstance(a, Agent)
	
	def test_aot_agent(self):
		np.random.seed(42)
		a = RandomAgent(1)
		state = Cube.scramble(10)
		action = a.act(state)
		if action is None:
			for action in a.searcher.action_queue:
				state = Cube.rotate(state, *action)
		else:
			while not Cube.is_solved(state):
				state = Cube.rotate(state, *action)
				action = a.act(state)
			
	def test_jit_agent(self):
		np.random.seed(42)
		a = PolicyCube.from_saved("local_train")
		state = Cube.scramble(10)
		for _ in range(10):
			state = Cube.rotate(state, *a.act(state))
