import numpy as np
import torch

from tests import MainTest

from librubiks import gpu
from librubiks import cube
from librubiks.model import Model, ModelConfig

from librubiks.solving.search import Searcher, RandomDFS, BFS, PolicySearch, ValueSearch, DankSearch, MCTS, AStar

def _action_queue_test(state, searcher, sol_found):
	for action in searcher.action_queue:
		state = cube.rotate(state, *cube.action_space[action])
	assert cube.is_solved(state) == sol_found

class TestSearchers(MainTest):
	def test_searchers(self):
		net = Model.create(ModelConfig())
		searchers = [
			RandomDFS(),
			BFS(),
			PolicySearch(net, sample_policy=False),
			PolicySearch(net, sample_policy=True),
			ValueSearch(net),
			DankSearch(net, 0.1, 4, 12),
		]
		for s in searchers: self._test_searcher(s)

	def _test_searcher(self, searcher: Searcher):
		state, _, _ = cube.scramble(4)
		solution_found  = searcher.search(state, .01)
		for action in searcher.action_queue:
			state = cube.rotate(state, *cube.action_space[action])
		assert solution_found == cube.is_solved(state)

class TestMCTS(MainTest):

	def test_search(self):
		state, _, _ = cube.scramble(50)
		self._mcts_test(state, False)
		state, _, _ = cube.scramble(3)
		searcher, sol_found = self._mcts_test(state, False)
		_action_queue_test(state, searcher, sol_found)
		searcher, sol_found = self._mcts_test(state, True)
		_action_queue_test(state, searcher, sol_found)

	def _mcts_test(self, state: np.ndarray, search_graph: bool):
		searcher = MCTS(Model.create(ModelConfig()), c=1, search_graph=search_graph)
		solved = searcher.search(state, .2)

		# Indices
		assert searcher.indices[state.tostring()] == 1
		for s, i in searcher.indices.items():
			assert searcher.states[i].tostring() == s
		assert sorted(searcher.indices.values())[0] == 1
		assert np.all(np.diff(sorted(searcher.indices.values())) == 1)

		used_idcs = np.array(list(searcher.indices.values()))

		# States
		assert np.all(searcher.states[1] == state)
		for i, s in enumerate(searcher.states):
			if i not in used_idcs: continue
			assert s.tostring() in searcher.indices
			assert searcher.indices[s.tostring()] == i

		# Neighbors
		if not search_graph:
			for i, neighs in enumerate(searcher.neighbors):
				if i not in used_idcs: continue
				state = searcher.states[i]
				for j, neighbor_index in enumerate(neighs):
					assert neighbor_index == 0 or neighbor_index in searcher.indices.values()
					if neighbor_index == 0: continue
					substate = cube.rotate(state, *cube.action_space[j])
					assert np.all(searcher.states[neighbor_index] == substate)

		# Policy and value
		with torch.no_grad():
			p, v = searcher.net(cube.as_oh(searcher.states[used_idcs]))
		p, v = p.softmax(dim=1).cpu().numpy(), v.squeeze().cpu().numpy()
		assert np.all(np.isclose(searcher.P[used_idcs], p, atol=1e-5))
		assert np.all(np.isclose(searcher.V[used_idcs], v, atol=1e-5))

		# Leaves
		if not search_graph:
			assert np.all(searcher.neighbors.all(axis=1) != searcher.leaves)

		# W
		assert searcher.W[used_idcs].all()

		return searcher, solved

class TestAStar(MainTest):

	#TODO: More indepth testing: Especially of updating of parents

	def test_search(self):
		test_params = {
			(0, 10),
			(0.5, 2),
			(1, 1),
		}
		net = Model.create(ModelConfig()).to(gpu).eval()
		for params in test_params:
			searcher = AStar(net, *params)
			self._can_win_all_easy_games(searcher)
			searcher.reset("Tue", "Herlau")
			assert not len(searcher.indices)
			assert not len(searcher.open_queue)
			assert not searcher.open_.any()

	def _can_win_all_easy_games(self, searcher):
		state, i, j = cube.scramble(2, force_not_solved=True)
		is_solved = searcher.search(state, time_limit=1)
		if is_solved:
			for action in searcher.action_queue:
				state = cube.rotate(state, *cube.action_space[action])
			assert cube.is_solved(state)

	def test_expansion(self):
		net = Model.create(ModelConfig()).to(gpu).eval()
		init_state, _, _ = cube.scramble(3)
		searcher = AStar(net, lambda_=0.1, expansions=5)
		searcher.search(init_state, time_limit=1)
		init_idx = searcher.indices[init_state.tostring()]
		assert init_idx == 1
		assert searcher.G[init_idx]  == 0
		assert searcher.closed[init_idx]
		for action in cube.action_space:
			substate = cube.rotate(init_state, *action)
			idx = searcher.indices[substate.tostring()]
			assert searcher.G[idx] == 1
			assert searcher.parents[idx] == init_idx
			assert searcher.cost(idx) is not None
	def test_batched_H(self):
		net = Model.create(ModelConfig()).to(gpu).eval()
		games = 5
		states, _ = cube.sequence_scrambler(games, 1, True)
		searcher = AStar(net, lambda_=1, expansions=2)
		J = searcher.batched_H(states)
		assert J.shape == (games,)



