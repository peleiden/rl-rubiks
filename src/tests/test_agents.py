import pytest
import numpy as np

from src.rubiks.cube.cube import Cube
from src.rubiks.solving.agents import Agent, RandomAgent


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
	def test_act(self):
		np.random.seed(42)
		a = RandomAgent(2)
		state = Cube.get_solved()
		for _ in range(10):
			state = Cube.rotate(state, *a.act(state))
