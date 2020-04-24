import numpy as np

from src.rubiks import gpu, set_repr
from src.rubiks.cube.cube import Cube
from src.rubiks.model import Model, ModelConfig
from src.rubiks.solving.search import MCTS
from src.rubiks.utils import seedsetter
from src.tests import MainTest


class TestMCTS(MainTest):
	
	def test_search(self):
		set_repr(False)
		net = Model.create(ModelConfig()).to(gpu).eval()
		state, _, _ = Cube.scramble(50)
		searcher = MCTS(net, c=1, nu=.1, search_graph=True, workers=10)
		
		# Generates a search tree and tests its correctness
		searcher.search(state, 1)
		for state in searcher.states.values():
			# Test neighbors and leaf status
			assert not state.is_leaf == all(state.neighs)
			for i, neigh in enumerate(state.neighs):
				new_state = Cube.rotate(state.state, *Cube.action_space[i])
				if neigh:
					assert neigh.state.tostring() == new_state.tostring()
					assert new_state.tostring() in searcher.states
				# else:
				# 	assert new_state.tostring() not in searcher.states
			# Assert that all W's are calculated correctly
			W = np.zeros(Cube.action_dim)
			for i, neigh in enumerate(state.neighs):
				if neigh and all(neigh.neighs):
					values = [x.value for x in neigh.neighs]
					W[i] = max(values)
			if (W != 0).all():
				assert all([x for neigh in state.neighs for x in neigh.neighs])
			if all(state.neighs) and all([all(neigh.neighs) for neigh in state.neighs if neigh]):
				assert (W != 0).all()
			assert (np.array(W) == state.W).all()
			# Tests P
			assert np.isclose(state.P.sum(), 1)
			set_repr(True)





