
import torch
from copy import deepcopy


class RubiksCube:

	def __init__(self, device = torch.device("cpu")):

		"""
		Shape: 6 x 8 uint8, see method three here: https://stackoverflow.com/a/55505784
		"""

		self.device = device
		self.state = torch.zeros(6, 8, dtype = torch.uint8, device = device)
		for i in range(6):
			self.state[i] = i
		
		# The i'th index contain the neighbors of the i'th side in positive direction
		# Do not make this a tensor, as it will slow execution significantly
		self.neighbors = (
			(1, 5, 4, 2),  # Front
			(2, 3, 5, 0),  # Left
			(0, 4, 3, 1),  # Top
			(5, 1, 2, 4),  # Back
			(3, 2, 0, 5),  # Right
			(4, 0, 1, 3),  # Bottom
		)
		# Du not make this a tuple, as it will slow execution significantly
		self.revolution = torch.tensor((
			(6, 7, 0),
			(2, 3, 4),
			(4, 5, 6),
			(0, 1, 2),
		), device = device)
	
	def rotate(self, face: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		# Leave this check out, as it increases runtime by ~15 %
		# if not 0 <= face <= 5:
		# 	raise IndexError("Face should be 0-5, not %i" % face)

		# Rolls the face
		shift = 1 if pos_rev else -1
		self.state[face] = self.state[face].roll(2 * shift)
		
		# Rolls the adjacent rows
		rowvec = torch.cat([
			self.state[self.neighbors[face][i], self.revolution[i]] for i in range(4)
		]).roll(3 * shift)
		for i in range(4):
			self.state[self.neighbors[face][i], self.revolution[i]] = rowvec[i*3:(i+1)*3]
		
	def __str__(self):

		return str(self.state)
	
	def to(self, device: torch.device, in_place = True):

		"""
		Mimicks tensors' to method
		Use only for device changing
		If not in_place, a new RubiksCube instance is returned
		"""

		self.device = device
		attrs = ("state", )
		new_rc = deepcopy(self)
		for attr in attrs:
			setattr(new_rc, attr, getattr(self, attr).to(device))
		
		if in_place:
			self = new_rc
		else:
			return new_rc
	
	def scramble(self, n: int):

		faces = torch.randint(6, (n, ))
		dirs = torch.randint(2, (n, ))

		for face, d in zip(faces, dirs):
			self.rotate(face, d)
		
		return faces, dirs
	
	def is_complete(self):
		 
		return bool(list(filter(lambda x: (x[0] == x).all(), self.state)))


if __name__ == "__main__":
	import numpy as np
	from utils.ticktock import TickTock
	n = int(1e4)
	tt = TickTock()

	# tt.tick()
	# cpu_rube = RubiksCube()
	# cpu_rube.scramble(n)
	# tt.tock()

	# if torch.cuda.is_available():
	# 	device = torch.device("cuda")
	# 	tt.tick()
	# 	gpu_rube = RubiksCube(device)
	# 	gpu_rube.scramble(n)
	# 	tt.tock()

	def test_scramble(_, device = torch.device("cpu")):
		rube = RubiksCube(device)
		rube.scramble(n)

	import multiprocessing as mp
	import matplotlib.pyplot as plt
	nps = range(1, 6)
	times = np.empty(nps.stop - nps.start)
	moves = np.empty(nps.stop - nps.start)
	games = 12
	for n_processes in nps:
		with mp.Pool(n_processes) as p:
			tt.tick()
			p.map(test_scramble, np.empty(games))
			times[n_processes-nps.start] = tt.tock(False)
			moves[n_processes-nps.start] = n * n_processes
	plt.plot(nps, times)
	plt.xlabel("Number of threads")
	plt.ylabel("Time to complete %i games of %i rotations" % (games, n))
	plt.grid(True)
	plt.show()

