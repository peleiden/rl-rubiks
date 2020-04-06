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
		state, _, _ = Cube.scramble(10)
		solution_found = a.generate_action_queue(state)
		if solution_found:
			while not Cube.is_solved(state):
				state = Cube.rotate(state, *a.act(state))
			assert Cube.is_solved(state)
		else:
			while a._searcher.action_queue:
				state = Cube.rotate(state, *a.act(state))
			assert not Cube.is_solved(state)
			
	def test_jit_agent(self):
		np.random.seed(42)
		a = PolicyCube.from_saved("src/rubiks/local_train")
		state, _, _ = Cube.scramble(10)
		for _ in range(10):
			state = Cube.rotate(state, *a.act(state))
