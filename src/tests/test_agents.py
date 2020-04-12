import os, sys
import pytest
import numpy as np
import torch

from src.rubiks.cube.cube import Cube
from src.rubiks.solving.agents import Agent, DeepAgent
from src.rubiks.solving.search import RandomDFS, BFS, PolicySearch, MCTS
from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.model import Model, ModelConfig
from src.rubiks.train import Train
from src.rubiks import cpu, gpu

def test_agents():

	path =  os.path.join("data", "hpc-20-04-12")
	agents = [
		Agent(RandomDFS()),
		Agent(BFS()),
		DeepAgent(PolicySearch.from_saved(path, False)),
		DeepAgent(PolicySearch.from_saved(path, True)),
		DeepAgent(MCTS.from_saved(path))
	]
	for agent in agents:
		_test_agent(agent)

def _test_agent(agent: Agent):
	state, _, _ = Cube.scramble(4)
	solution_found, steps = agent.generate_action_queue(state, .01)
	for _ in range(steps):
		state = Cube.rotate(state, *agent.action())
	assert solution_found == Cube.is_solved(state)
