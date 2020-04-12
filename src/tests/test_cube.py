from src.tests.main import MainTest

from src.rubiks import get_repr, set_repr
from src.rubiks.cube.cube import Cube
from src.rubiks.cube.maps import SimpleState, get_corner_pos, get_side_pos


import numpy as np
import torch

class TestRubiksCube(MainTest):
	def test_init(self):
		state = Cube.get_solved()
		assert Cube.is_solved(state)
		assert Cube.get_solved_instance().shape == (20,)

	def test_cube(self):
		set_repr(True)
		self._rotation_tests()
		# TODO Once _Cube686.as633 is implemented
		set_repr(False)
		self._rotation_tests()
		set_repr(True)

	def _rotation_tests(self):
		state = Cube.get_solved()
		for action in Cube.action_space:
			state = Cube.rotate(state, *action)
		# Tests that stringify and by extensions as633 works on assembled
		state = Cube.get_solved()
		assert Cube.stringify(state) == "\n".join([
			"      2 2 2            ",
			"      2 2 2            ",
			"      2 2 2            ",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"      3 3 3            ",
			"      3 3 3            ",
			"      3 3 3            ",
		])
		# Performs moves and checks if are assembled/not assembled as expected
		moves = ((0, 1), (0, 0), (0, 1), (1, 1), (2, 0), (3, 0))
		assembled = (False, True, False, False, False, False)
		for m, a in zip(moves, assembled):
			state = Cube.rotate(state, *m)
			assert a == Cube.is_solved(state)

		# Tests more moves
		moves = ((3, 1), (2, 1), (1, 0), (0, 0))
		assembled = (False, False, False, True)
		for m, a in zip(moves, assembled):
			state = Cube.rotate(state, *m)
			assert a == Cube.is_solved(state)

		# Performs move and checks if it fits with how the string representation would look
		state = Cube.get_solved()
		state = Cube.rotate(state, *(0, 1))
		assert Cube.stringify(state) == "\n".join([
			"      2 2 2            ",
			"      2 2 2            ",
			"      5 5 5            ",
			"4 4 2 0 0 0 3 5 5 1 1 1",
			"4 4 2 0 0 0 3 5 5 1 1 1",
			"4 4 2 0 0 0 3 5 5 1 1 1",
			"      4 4 4            ",
			"      3 3 3            ",
			"      3 3 3            ",
		])

		# Performs all moves and checks if result fits with how it theoretically should look
		state = Cube.get_solved()
		moves = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
				 (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1))
		assembled = (False, False, False, False, False, False,
					 False, False, False, False, False, False)
		for m, a in zip(moves, assembled):
			state = Cube.rotate(state, *m)
			assert a == Cube.is_solved(state)
		assert Cube.stringify(state) == "\n".join([
			"      2 0 2            ",
			"      5 2 4            ",
			"      2 1 2            ",
			"4 2 4 0 2 0 5 2 5 1 2 1",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"4 3 4 0 3 0 5 3 5 1 3 1",
			"      3 1 3            ",
			"      5 3 4            ",
			"      3 0 3            ",
		])

	def test_scramble(self):
		np.random.seed(42)
		state = Cube.get_solved()
		state, faces, dirs = Cube.scramble(1)
		assert not Cube.is_solved(state)

		state = Cube.get_solved()
		state, faces, dirs = Cube.scramble(20)
		assert not Cube.is_solved(state)
		for f, d in zip(reversed(faces), reversed([not item for item in dirs])):
			state = Cube.rotate(state, *(f, d))
		assert Cube.is_solved(state)

	def test_as_oh(self):
		state = Cube.get_solved()
		oh = Cube.as_oh(state)
		supposed_state = torch.zeros(20, 24)
		corners = [get_corner_pos(c, o) for c, o in zip(SimpleState.corners, SimpleState.corner_orientations)]
		supposed_state[torch.arange(8), corners] = 1
		sides = [get_side_pos(s, o) for s, o in zip(SimpleState.sides, SimpleState.side_orientations)]
		supposed_state[torch.arange(8, 20), sides] = 1
		assert (supposed_state.flatten() == oh).all()

	def test_as633(self):
		state = Cube.as633(Cube.get_solved())
		target633 = list()
		for i in range(6):
			target633.append(np.ones((3, 3)) * i)
		target633 = np.array(target633)
		assert (state == target633).all()




	# def test_reset(self):
	# 	r = Cube()
	# 	state = get_assembled()
	# 	np.random.seed(42)
	# 	N = r.reset()
	# 	assert not r.is_assembled()
	# 	assert N >= r.scrambling_procedure['N_scrambles'][0]
	# 	assert N < r.scrambling_procedure['N_scrambles'][1]

