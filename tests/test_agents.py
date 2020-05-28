import os, sys
import pytest
import numpy as np
import torch

from tests import MainTest

from librubiks.cube import Cube
from librubiks.model import Model, ModelConfig
from librubiks.solving.agents import Agent, DeepAgent
from librubiks.solving.search import RandomDFS, BFS, PolicySearch, MCTS
from librubiks import gpu

class TestAgent(MainTest):
	def test_agents(self):

		net = Model.create(ModelConfig())
		agents = [
			Agent(RandomDFS()),
			Agent(BFS()),
			DeepAgent(PolicySearch(net, sample_policy=False)),
			DeepAgent(PolicySearch(net, sample_policy=True)),
			DeepAgent(MCTS(net, c=1, search_graph=False))
		]
		for agent in agents:
			self._test_agent(agent)

	def _test_agent(self, agent: Agent):
		state, _, _ = Cube.scramble(4)
		solution_found, steps = agent.generate_action_queue(state, .01)
		for action in agent.actions():
			state = Cube.rotate(state, *action)
		assert solution_found == Cube.is_solved(state)
