# Hacky import code needed to import from sibling dir
import sys, os
sys.path.append(os.path.join(sys.path[0], '..', 'src'))

import torch
from rubiks import RubiksCube

class TestCube:
	def test_cube(self, n = 5):
		cube = RubiksCube(n)
		print(cube.state.shape, cube.state.size == torch.Size((6, n, n)))
		assert cube.state.shape == torch.Size((6, n, n))

