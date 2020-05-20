import os, sys
import pytest
import numpy as np
import torch

from tests import MainTest

from librubiks.cube import Cube

from librubiks.solving.agents import Agent, DeepAgent
from librubiks.solving.search import RandomDFS, BFS, PolicySearch, MCTS
from librubiks import gpu

class TestAgent(MainTest):
	def test_agents(self):

		path =  os.path.join("data", "hpc-20-04-12")
		agents = [
			Agent(RandomDFS()),
			Agent(BFS()),
			DeepAgent(PolicySearch.from_saved(path, False)),
			DeepAgent(PolicySearch.from_saved(path, True)),
			DeepAgent(MCTS.from_saved(path, use_best=1, c=1, nu=True, search_graph=True, workers=1, policy_type='p'))
		]
		for agent in agents:
			self._test_agent(agent)

	def _test_agent(self, agent: Agent):
		state, _, _ = Cube.scramble(4)
		solution_found, steps = agent.generate_action_queue(state, .01)
		for action in agent.actions():
			state = Cube.rotate(state, *action)
		assert solution_found == Cube.is_solved(state)
