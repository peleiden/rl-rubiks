import os, sys
import pytest
import numpy as np

from src.rubiks.cube.cube import Cube
from src.rubiks.solving.agents import Agent, DeepAgent
from src.rubiks.solving.search import RandomDFS, BFS, PolicySearch, MCTS

def test_agents():
	path = os.path.join(sys.path[0], "src", "rubiks", "local_train")
	agents = [
		Agent(RandomDFS()),
		DeepAgent(PolicySearch.from_saved(path, False)),
		DeepAgent(PolicySearch.from_saved(path, True)),
		DeepAgent(MCTS.from_saved(path))
	]

def _test_agent(agent: Agent):
	state, _, _ = Cube.scramble(4)
	solution_found, steps = agent.generate_action_queue(state, .01)
	for _ in range(steps):
		state = Cube.rotate(state, *Cube.action_space[agent.action()])
	assert solution_found == Cube.is_solved(state)