import numpy as np
import torch

from tests import MainTest

from librubiks import gpu, cube
from librubiks.cube import with_used_repr
from librubiks.cube.maps import SimpleState, get_corner_pos, get_side_pos

class TestRubiksCube(MainTest):
	is2024: bool

	def test_init(self):
		state = cube.get_solved()
		assert cube.is_solved(state)
		assert cube.get_solved_instance().shape == (20,)

	def test_cube(self):
		self.is2024 = True
		self._rotation_tests()
		self._multi_rotate_test()
		self.is2024 = False
		self._rotation_tests()
		self._multi_rotate_test()

	@with_used_repr
	def _rotation_tests(self):
		state = cube.get_solved()
		for action in cube.action_space:
			state = cube.rotate(state, *action)
		# Tests that stringify and by extensions as633 works on assembled
		state = cube.get_solved()
		assert cube.stringify(state) == "\n".join([
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
			state = cube.rotate(state, *m)
			assert a == cube.is_solved(state)

		# Tests more moves
		moves = ((3, 1), (2, 1), (1, 0), (0, 0))
		assembled = (False, False, False, True)
		for m, a in zip(moves, assembled):
			state = cube.rotate(state, *m)
			assert a == cube.is_solved(state)

		# Performs move and checks if it fits with how the string representation would look
		state = cube.get_solved()
		state = cube.rotate(state, *(0, 1))
		assert cube.stringify(state) == "\n".join([
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
		state = cube.get_solved()
		moves = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
				 (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1))
		assembled = (False, False, False, False, False, False,
					 False, False, False, False, False, False)
		for m, a in zip(moves, assembled):
			state = cube.rotate(state, *m)
			assert a == cube.is_solved(state)
		assert cube.stringify(state) == "\n".join([
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

	@with_used_repr
	def _multi_rotate_test(self):
		states = np.array([cube.get_solved()]*5)
		for _ in range(10):
			faces, dirs = np.random.randint(0, 6, 5), np.random.randint(0, 1, 5)
			states_classic = np.array([cube.rotate(state, face, d) for state, face, d in zip(states, faces, dirs)])
			states = cube.multi_rotate(states, faces, dirs)
			assert (states_classic == states).all()

	def test_scramble(self):
		np.random.seed(42)
		state = cube.get_solved()
		state, faces, dirs = cube.scramble(1)
		assert not cube.is_solved(state)

		state = cube.get_solved()
		state, faces, dirs = cube.scramble(20)
		assert not cube.is_solved(state)
		for f, d in zip(reversed(faces), reversed([int(not item) for item in dirs])):
			state = cube.rotate(state, *(f, d))
		assert cube.is_solved(state)

	def test_iter_actions(self):
		actions = np.array([
			[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 2,
			[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2,
		], dtype=np.uint8)
		assert np.all(actions==cube.iter_actions(2))

	def test_indices_to_actions(self):
		actions = np.arange(cube.action_dim)
		faces, dirs = cube.indices_to_actions(actions)
		assert np.all(faces == np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))
		assert np.all(dirs == np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))

	def test_as_oh(self):
		state = cube.get_solved()
		oh = cube.as_oh(state)
		supposed_state = torch.zeros(20, 24, device=gpu)
		corners = [get_corner_pos(c, o) for c, o
				   in zip(SimpleState.corners.tolist(), SimpleState.corner_orientations.tolist())]
		supposed_state[torch.arange(8), corners] = 1
		sides = [get_side_pos(s, o) for s, o
				 in zip(SimpleState.sides.tolist(), SimpleState.side_orientations.tolist())]
		supposed_state[torch.arange(8, 20), sides] = 1
		assert (supposed_state.flatten() == oh).all()

	def test_as633(self):
		state = cube.as633(cube.get_solved())
		target633 = list()
		for i in range(6):
			target633.append(np.ones((3, 3)) * i)
		target633 = np.array(target633)
		assert (state == target633).all()

	def test_correctness(self):
		self.is2024 = False
		self._get_correctness()

	@with_used_repr
	def _get_correctness(self):
		state = cube.get_solved()
		state = cube.rotate(state, 0, True)
		state = cube.rotate(state, 5, False)
		correctness = torch.tensor([
			[1, 1, 1, 1, -1, -1, -1, 1],
			[-1, 1, 1, 1, 1, 1, -1, -1],
			[-1, -1, -1, -1, -1, 1, 1, 1],
			[-1, -1, -1, -1, -1, 1, 1, 1],
			[-1, 1, 1, 1, 1, 1, -1, -1],
			[1, 1, -1, -1, -1, 1, 1, 1],
		], device=gpu)
		assert torch.all(correctness == cube.as_correct(torch.from_numpy(state).unsqueeze(0)))

