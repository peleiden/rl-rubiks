# Hacky import code needed to import from sibling dir
import sys, os
from ..rubiks.rubiks import RubiksCube

import torch

class TestCube:
	pass
	# def test_cube(self, n = 5):
	# 	cube = RubiksCube(n)
	# 	print(cube.state.shape, cube.state.size == torch.Size((6, n, n)))
	# 	assert cube.state.shape == torch.Size((6, n, n))

