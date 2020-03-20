import pytest
import numpy as np

from src.rubiks.cube.cube import Cube
from src.rubiks.post_train.agents import Agent, RandomAgent


class TestAgent:
	
	def test_actions(self):
		a = Agent()
		r = Cube()
		# Make sure every aciton in action space is possible
		for action in a.action_space:
			r.move(*action)
		assert a.action_dim == 12
		
	def test_act(self):
		a = Agent()
		with pytest.raises(NotImplementedError) as e_info:
			a.act(None)

	def test_model(self):
		a = Agent(model_needed=True)
		assert a.model_env == Cube

class TestRandomAgent:
	
	def test_init(self):
		a = RandomAgent()
		assert isinstance(a, Agent)
		assert hasattr(a, 'model_needed')
		assert a.model_env is None

	def test_act(self):
		np.random.seed(42)
		a = RandomAgent()
		r = Cube()

		for _ in range(10):
			r.move(*a.act(None))
