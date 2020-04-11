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

	net = Model(ModelConfig()).to(gpu)
	evaluator = Evaluator(2, 2, [2])
	net = Train(rollouts=1, batch_size=2, rollout_games=2, rollout_depth=3, optim_fn=torch.optim.Adam, searcher_class=PolicySearch, lr=1e-6, evaluations=1, evaluator=evaluator).train(net)
	path = os.path.join(sys.path[0], "src", "rubiks", "local_train")
	net.save(path)
	agents = [
		Agent(RandomDFS()),
		Agent(BFS()),
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
