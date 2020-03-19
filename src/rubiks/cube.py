import numpy as np
import torch


class RubiksCube:

	assembled = torch.zeros(20)

	# Scrambling procedure saved as dict for reproducability 
	scrambling_procedure = {
		'N_scrambles':	(5, 10),  # Tuple for scrambling random # moves in uniform [low, high[
	}

	action_space = list()
	for i in range(6): action_space.extend( [(i, True), (i, False)] )
	action_dim = len(action_space)

	def __init__(self):

		"""
		Shape: 6 x 8 x 6 one hot encoding - method three here: https://stackoverflow.com/a/55505784 
		"""

		self.state = self.assembled.copy()

	# @staticmethod
	# def as_tensor(state: State):
	# 	# TODO: Device here?
	# 	t = torch.zeros(20, 24)
	# 	onepos = torch.empty(20).long()
	# 	onepos[:8] = state.corners * 3 + state.corner_orientations
	# 	onepos[8:] = state.sides * 2 + state.side_orientations
	# 	t[torch.arange(20).long(), onepos] = 1
	# 	return t

	def move(self, face: int, pos_rev: bool):
		"""
		Performs rotation, mutates state and returns whether cube is completed
		"""
		self.state = self.rotate(self.state, face, pos_rev)
		return self.is_assembled()
		
	def reset(self):
		"""
		Resets cube by random scramblings in accordance with self.scrambling_procedure
		"""
		self.state = self.assembled.copy()		
		N_resets = np.random.randint(*self.scrambling_procedure["N_scrambles"])
		self.scramble(N_resets)
		
		while self.is_assembled(): 
			self.scramble(1) # Avoid randomly solving the cube
			N_resets += 1
		return N_resets
	
	def rotate(self, current_state: np.array, face: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		# if not 0 <= face <= 5:
		# 	raise IndexError("Face should be 0-5, not %i" % face)
		altered_state = current_state.copy()

		altered_state[face] = self._shift_right(self.state[face], 2)\
			if pos_rev else self._shift_left(self.state[face], 2)
		
		ini_state = current_state[self.neighbours[face]]
		
		if pos_rev:
			for i in range(4):
				altered_state[self.neighbours[face, i], self.adjacents[i]]\
					= ini_state[i-1, self.adjacents[i-1]]
		else:
			for i in range(4):
				altered_state[self.neighbours[face, i-1], self.adjacents[i-1]]\
					= ini_state[i, self.adjacents[i]]
		
		return altered_state
		
	@staticmethod
	def _shift_left(a: np.ndarray, num_elems: int):

		return np.roll(a, -num_elems, axis = 0)
	
	@staticmethod
	def _shift_right(a: np.ndarray, num_elems: int):

		return np.roll(a, num_elems, axis = 0)

	def scramble(self, n: int):

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)

		for face, d in zip(faces, dirs):
			self.state = self.rotate(self.state, face, d)  # Uses rotate instead of move as checking for victory is not needed here
		
		return faces, dirs
	
	def sequence_scrambler(self, n: int):
		"""
		A non-inplace scrambler which returns the state to each of the scrambles useful for ADI
		"""
		scrambled_states = np.empty((n, *self.assembled.shape))

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)

		scrambled_states[0] = self.assembled
		for i, face, d in zip(range(n-1), faces, dirs):
			scrambled_states[i+1] = self.rotate(scrambled_states[i], face, d)
		return scrambled_states


	def is_assembled(self):
		
		return (self.state == self.assembled).all()
		
	def __str__(self):
		
		return str(self.as68())
	
	def as68(self):

		"""
		Un-encodes one-hot and returns self.state as 6x8 matrix 
		"""

		state68 = np.where(self.state == 1)[2].reshape((6, 8))
		return state68

if __name__ == "__main__":
	
	# Benchmarking example
	from src.rubiks.utils.benchmark import Benchmark
	def test_scramble(games):
		# Function is weird, as it is designed to work for both single and multithreaded benchmarks
		if hasattr(games, "__iter__"):
			for _ in games:
				rube = RubiksCube()
				rube.scramble(n)
		else:
			rube = RubiksCube()
			rube.scramble(n)

	n = int(1e4)
	nt = range(1, 5)
	games = np.empty(12)

	title = f"Scramble bench: {games.size} cubes each with {n} scrambles"
	bm = Benchmark(test_scramble, "local_benchmarks/scramble_example", title)
	bm.singlethreaded("", games)
	threads, times = bm.multithreaded(nt, games)
	bm.plot_mt_results(threads, times, title)

	
	

