# Hacky import code needed to import from sibling dir
import sys, os
from ..rubiks.cube import RubiksCube

import torch
import numpy as np

class TestRubiksCube:
	def test_init(self):
		r = RubiksCube()
		assert r.is_assembled()
		assert r.state.shape == (6, 8, 6)
		assert r.as68().shape == (6, 8)
	def test_move(self):
		r = RubiksCube()
		assert not r.move(0, 1)
		assert r.move(0, 0)

		assert not r.move(0, 1)
		assert not r.move(1, 1)
		assert not r.move(2, 0)
		assert not r.move(3, 0)

		assert np.all (
		r.as68() == np.array([[5, 5, 5, 0, 0, 0, 4, 4],
		[0, 0, 0, 0, 0, 1, 1, 1],
		[4, 2, 2, 2, 4, 4, 4, 4],
		[2, 2, 2, 2, 2, 3, 3, 3],
		[3, 5, 5, 3, 3, 5, 5, 4],
		[1, 5, 1, 1, 3, 3, 1, 1],
		])
		)

		assert not r.move(3, 1)
		assert not r.move(2, 1)
		assert not r.move(1, 0)
		assert r.move(0, 0)

	def test_reset(self):
		r = RubiksCube()
		r.reset()
		assert not r.is_assembled()
	
	def test_scramble(self):
		r = RubiksCube()
		np.random.seed(42)

		N = 20
		faces, dirs = r.scramble(N)
		for face, direc in zip(reversed(faces), reversed(dirs)) :
			r.move( face, not direc )

		assert r.is_assembled()

	def test_as68(self):
		np.random.seed(42)

		 
		r = RubiksCube()

		cube_68 = r.as68()
		cube_68[1, 2] = 1
		cube_68[2, 2] = 3

		r.state[1, 2] = 0
		r.state[1, 2, 1] = 1
		r.state[2, 2] = 0
		r.state[2, 2, 3] =1


		assert np.all(r.as68() == cube_68)

