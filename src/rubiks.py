import numpy as np


class RubiksCube:

	assembled = np.empty((6, 8), dtype = np.uint8)
	for i in range(6):
		assembled[i] = i

	def __init__(self):

		"""
		Shape: 6 x 8 uint8, see method three here: https://stackoverflow.com/a/55505784
		"""

		self.state = self.assembled.copy()
		
		# The i'th index contain the neighbors of the i'th side in positive direction
		self.neighbors = np.array([
			[1, 5, 4, 2],  # Front
			[2, 3, 5, 0],  # Left
			[0, 4, 3, 1],  # Top
			[5, 1, 2, 4],  # Back
			[3, 2, 0, 5],  # Right
			[4, 0, 1, 3],  # Bottom
		])
		self.adjecents = np.array([
			[6, 7, 0],
			[2, 3, 4],
			[4, 5, 6],
			[0, 1, 2],
		])
	
	@staticmethod
	def _shift_left(a: np.ndarray, bits: int):

		return np.roll(a, -bits)
	
	@staticmethod
	def _shift_right(a: np.ndarray, bits: int):

		return np.roll(a, bits)
	
	def rotate(self, face: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		# if not 0 <= face <= 5:
		# 	raise IndexError("Face should be 0-5, not %i" % face)

		self.state[face] = self._shift_right(self.state[face], 2)\
			if pos_rev else self._shift_left(self.state[face], 2)
		
		ini_state = self.state[self.neighbors[face]].copy()
		if pos_rev:
			for i in range(4):
				self.state[self.neighbors[face][i], self.adjecents[i]]\
					= ini_state[i-1, self.adjecents[i-1]]
		else:
			for i in range(4):
				self.state[self.neighbors[face][i-1], self.adjecents[i-1]]\
					= ini_state[i, self.adjecents[i]]

	def scramble(self, n: int):

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)

		for face, d in zip(faces, dirs):
			self.rotate(face, d)
		
		return faces, dirs
	
	def is_complete(self):
		
		return (self.state == self.assembled).all()
	
	# def to(device: torch.device)
		
	def __str__(self):
		
		return str(self.state)


if __name__ == "__main__":
	from utils.ticktock import TickTock
	n = int(1e5)
	tt = TickTock()
	# rube = RubiksCube()
	# tt.tick()
	# rube.scramble(n)
	# tt.tock()

	def test_scramble(_):
		rube = RubiksCube()
		rube.scramble(n)

	import multiprocessing as mp
	import matplotlib.pyplot as plt
	nps = range(5, 13)
	times = np.empty(nps.stop - nps.start)
	moves = np.empty(nps.stop - nps.start)
	games = 24
	for n_processes in nps:
		with mp.Pool(n_processes) as p:
			tt.tick()
			p.map(test_scramble, np.empty(games))
			times[n_processes-nps.start] = tt.tock(False)
			moves[n_processes-nps.start] = n * n_processes
			print(f"{games:02} games\n{n} moves per game\n{n_processes} threads\n{times[n_processes-nps.start]:4} seconds\n")
	plt.plot(nps, times)
	plt.xlabel("Number of threads")
	plt.ylabel("Time to complete %i games of %i rotations" % (games, n))
	plt.grid(True)
	plt.show()

	
	

