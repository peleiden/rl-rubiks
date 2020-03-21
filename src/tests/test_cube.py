# Hacky import code needed to import from sibling dir
from src.rubiks.cube.cube import Cube
from src.rubiks.cube.tensor_maps import SimpleState, get_corner_pos, get_side_pos

import numpy as np
import torch

class TestRubiksCube:
	def test_init(self):
		state = Cube.get_assembled()
		assert Cube.is_assembled(state)
		assert Cube.assembled.shape == (20,)
		
	def test_cube(self):
		# Tests that stringify and by extensions as633 works on assembled
		state = Cube.get_assembled()
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
			assert a == Cube.is_assembled(state)
		# TODO: Test that state is in expected state
		# assert Cube.stringify(state) == "\n".join([
		# 	"      2 2 2            ",
		# 	"      2 2 2            ",
		# 	"      2 2 2            ",
		# 	"4 4 4 0 0 0 5 5 5 1 1 1",
		# 	"4 4 4 0 0 0 5 5 5 1 1 1",
		# 	"4 4 4 0 0 0 5 5 5 1 1 1",
		# 	"      3 3 3            ",
		# 	"      3 3 3            ",
		# 	"      3 3 3            ",
		# ])
		# Tests more moves
		moves = ((3, 1), (2, 1), (1, 0), (0, 0))
		assembled = (False, False, False, True)
		for m, a in zip(moves, assembled):
			state = Cube.rotate(state, *m)
			assert a == Cube.is_assembled(state)
	
	def test_oh(self):
		state = Cube.get_assembled()
		oh = Cube.as_oh(state)
		supposed_state = torch.zeros(20, 24)
		corners = [get_corner_pos(c, o) for c, o in zip(SimpleState.corners, SimpleState.corner_orientations)]
		supposed_state[torch.arange(8), corners] = 1
		sides = [get_side_pos(s, o) for s, o in zip(SimpleState.sides, SimpleState.side_orientations)]
		supposed_state[torch.arange(8, 20), sides] = 1
		assert (supposed_state.flatten() == oh).all()
	
	def test_633(self):
		state = Cube.as633(Cube.get_assembled())
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
	
	# def test_scramble(self):
	# 	r = Cube()
	# 	np.random.seed(42)
	#
	# 	N = 20
	# 	faces, dirs = r.scramble(N)
	# 	for face, direc in zip(reversed(faces), reversed(dirs)) :
	# 		r.move( face, not direc )
	#
	# 	assert r.is_assembled()

