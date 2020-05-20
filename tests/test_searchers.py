import numpy as np
import torch

from tests import MainTest

from librubiks import gpu
from librubiks.cube import Cube
from librubiks.model import Model, ModelConfig
from librubiks.solving.search import MCTS, AStar


def _action_queue_test(state, searcher, sol_found):
	for action in searcher.action_queue:
		state = Cube.rotate(state, *Cube.action_space[action])
	assert Cube.is_solved(state) == sol_found


class TestMCTS(MainTest):

	def test_search(self):
		state, _, _ = Cube.scramble(50)
		self._mcts_test(state, False)
		state, _, _ = Cube.scramble(3)
		searcher, sol_found = self._mcts_test(state, False)
		_action_queue_test(state, searcher, sol_found)
		searcher, sol_found = self._mcts_test(state, True)
		_action_queue_test(state, searcher, sol_found)

	def _mcts_test(self, state: np.ndarray, search_graph: bool):
		searcher = MCTS.from_saved("data/hpc-20-04-12", False, c=1, nu=.001, search_graph=search_graph, workers=10, policy_type="p")
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
					substate = Cube.rotate(state, *Cube.action_space[j])
					assert np.all(searcher.states[neighbor_index] == substate)

		# Policy and value
		with torch.no_grad():
			p, v = searcher.net(Cube.as_oh(searcher.states[used_idcs]))
		p, v = p.softmax(dim=1).cpu().numpy(), v.squeeze().cpu().numpy()
		assert np.all(np.isclose(searcher.P[used_idcs], p, atol=1e-5))
		assert np.all(np.isclose(searcher.V[used_idcs], v, atol=1e-5))

		# Leaves
		if not search_graph:
			assert np.all(searcher.neighbors.all(axis=1) != searcher.leaves)

		# W
		for i in used_idcs:
			neighs = searcher.neighbors[i]
			supposed_Ws = np.zeros(Cube.action_dim)
			for j, neighbor_index in enumerate(neighs):
				if neighbor_index == 0: continue
				neighbor_neighbor_indices = searcher.neighbors[neighbor_index]
				if np.all(neighbor_neighbor_indices):
					supposed_Ws[j] = np.max(searcher.V[neighbor_neighbor_indices])
			assert np.all(supposed_Ws == searcher.W[i])
		
		return searcher, solved

class TestAStar(MainTest):

	def test_neighbors(self):
		net = Model.create(ModelConfig()).to(gpu).eval()
		state, _, _ = Cube.scramble(50)
		searcher = AStar(net)
		neighbors = searcher.get_neighbors(state)
		assert len(neighbors) == 12
