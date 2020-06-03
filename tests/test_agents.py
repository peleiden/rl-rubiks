import numpy as np
import torch

from tests import MainTest

from librubiks import gpu
from librubiks import cube
from librubiks.model import Model, ModelConfig

from librubiks.solving.agents import Agent, RandomSearch, BFS, PolicySearch, ValueSearch, EGVM, MCTS, AStar

def _action_queue_test(state, agent, sol_found):
	for action in agent.action_queue:
		state = cube.rotate(state, *cube.action_space[action])
	assert cube.is_solved(state) == sol_found

class TestAgents(MainTest):
	def test_agents(self):
		net = Model.create(ModelConfig())
		agents = [
			RandomSearch(),
			BFS(),
			PolicySearch(net, sample_policy=False),
			PolicySearch(net, sample_policy=True),
			ValueSearch(net),
			EGVM(net, 0.1, 4, 12),
		]
		for s in agents: self._test_agents(s)

	def _test_agents(self, agent: Agent):
		state, _, _ = cube.scramble(4)
		solution_found  = agent.search(state, .01)
		for action in agent.action_queue:
			state = cube.rotate(state, *cube.action_space[action])
		assert solution_found == cube.is_solved(state)

class TestMCTS(MainTest):

	def test_agent(self):
		state, _, _ = cube.scramble(50)
		self._mcts_test(state, False)
		state, _, _ = cube.scramble(3)
		agent, sol_found = self._mcts_test(state, False)
		_action_queue_test(state, agent, sol_found)
		agent, sol_found = self._mcts_test(state, True)
		_action_queue_test(state, agent, sol_found)

	def _mcts_test(self, state: np.ndarray, search_graph: bool):
		agent = MCTS(Model.create(ModelConfig()), c=1, search_graph=search_graph)
		solved = agent.search(state, .2)

		# Indices
		assert agent.indices[state.tostring()] == 1
		for s, i in agent.indices.items():
			assert agent.states[i].tostring() == s
		assert sorted(agent.indices.values())[0] == 1
		assert np.all(np.diff(sorted(agent.indices.values())) == 1)

		used_idcs = np.array(list(agent.indices.values()))

		# States
		assert np.all(agent.states[1] == state)
		for i, s in enumerate(agent.states):
			if i not in used_idcs: continue
			assert s.tostring() in agent.indices
			assert agent.indices[s.tostring()] == i

		# Neighbors
		if not search_graph:
			for i, neighs in enumerate(agent.neighbors):
				if i not in used_idcs: continue
				state = agent.states[i]
				for j, neighbor_index in enumerate(neighs):
					assert neighbor_index == 0 or neighbor_index in agent.indices.values()
					if neighbor_index == 0: continue
					substate = cube.rotate(state, *cube.action_space[j])
					assert np.all(agent.states[neighbor_index] == substate)

		# Policy and value
		with torch.no_grad():
			p, v = agent.net(cube.as_oh(agent.states[used_idcs]))
		p, v = p.softmax(dim=1).cpu().numpy(), v.squeeze().cpu().numpy()
		assert np.all(np.isclose(agent.P[used_idcs], p, atol=1e-5))
		assert np.all(np.isclose(agent.V[used_idcs], v, atol=1e-5))

		# Leaves
		if not search_graph:
			assert np.all(agent.neighbors.all(axis=1) != agent.leaves)

		# W
		assert agent.W[used_idcs].all()

		return agent, solved

class TestAStar(MainTest):

	#TODO: More indepth testing: Especially of updating of parents

	def test_agent(self):
		test_params = {
			(0, 10),
			(0.5, 2),
			(1, 1),
		}
		net = Model.create(ModelConfig()).eval()
		for params in test_params:
			agent = AStar(net, *params)
			self._can_win_all_easy_games(agent)
			agent.reset("Tue", "Herlau")
			assert not len(agent.indices)
			assert not len(agent.open_queue)

	def _can_win_all_easy_games(self, agent):
		state, i, j = cube.scramble(2, force_not_solved=True)
		is_solved = agent.search(state, time_limit=1)
		if is_solved:
			for action in agent.action_queue:
				state = cube.rotate(state, *cube.action_space[action])
			assert cube.is_solved(state)

	def test_expansion(self):
		net = Model.create(ModelConfig()).eval()
		init_state, _, _ = cube.scramble(3)
		agent = AStar(net, lambda_=0.1, expansions=5)
		agent.search(init_state, time_limit=1)
		init_idx = agent.indices[init_state.tostring()]
		assert init_idx == 1
		assert agent.G[init_idx]  == 0
		for action in cube.action_space:
			substate = cube.rotate(init_state, *action)
			idx = agent.indices[substate.tostring()]
			assert agent.G[idx] == 1
			assert agent.parents[idx] == init_idx

	def test_cost(self):
		net = Model.create(ModelConfig()).eval()
		games = 5
		states, _ = cube.sequence_scrambler(games, 1, True)
		agent = AStar(net, lambda_=1, expansions=2)
		agent.reset(1, 1)
		i = []
		for i, _ in enumerate(states): agent.G[i] = 1
		cost = agent.cost(states, i)
		assert cost.shape == (games,)



