import numpy as np
import torch
from src.rubiks.cube.tensor_map import SimpleState, get_corner_pos, get_side_pos, get_tensor_map

def _get_assembled(dtype=np.int8):
	assembled_state = SimpleState()
	tensor_state = np.empty(20, dtype=dtype)
	for i in range(8):
		tensor_state[i] = get_corner_pos(assembled_state.corners[i], assembled_state.corner_orientations[i])
	for i in range(12):
		tensor_state[i+8] = get_side_pos(assembled_state.sides[i], assembled_state.side_orientations[i])
	return tensor_state

class Cube:

	_assembled = _get_assembled()
	map_pos, map_neg = get_tensor_map()

	action_space = list()
	for i in range(6): action_space.extend( [(i, True), (i, False)] )
	action_dim = len(action_space)
	
	# The values for stickers on the corners in the order of their priority
	# The first is the tracked one
	corner_values = (
		(0, 2, 4),
		(0, 3, 4),
		(0, 3, 5),
		(0, 2, 5),
		(1, 2, 4),
		(1, 3, 4),
		(1, 3, 5),
		(1, 2, 5),
	)
	side_values = (
		(0, 2),
		(0, 4),
		(0, 3),
		(0, 5),
		(2, 4),
		(3, 4),
		(3, 5),
		(2, 5),
		(1, 2),
		(1, 4),
		(1, 3),
		(1, 5),
	)
	
	# NB: If the six sides are represented by an array, the order should be F, B, T, D, L, R
	# Corresponding colours
	colours = ["yellow", "white", "red", "orange", "blue", "green"]
	
	@classmethod
	def rotate(cls, current_state: np.ndarray, face: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""
		
		assert current_state.size == 20

		altered_state = current_state.copy()
		map_ = cls.map_pos[face] if pos_rev else cls.map_neg[face]
		altered_state[:8] += map_[0, altered_state[:8]]
		altered_state[8:] += map_[1, altered_state[8:]]
		
		return altered_state

	@classmethod
	def scramble(cls, n: int):

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)
		state = cls._assembled.copy()

		for face, d in zip(faces, dirs):
			state = cls.rotate(state, face, d)  # Uses rotate instead of move as checking for victory is not needed here
		
		return state, faces, dirs
	
	def sequence_scrambler(self, n: int):
		"""
		A non-inplace scrambler which returns the state to each of the scrambles useful for ADI
		"""
		scrambled_states = np.empty((n, *self._assembled.shape))

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)

		scrambled_states[0] = self._assembled
		for i, face, d in zip(range(n-1), faces, dirs):
			scrambled_states[i+1] = self.rotate(scrambled_states[i], face, d)
		return scrambled_states
	
	@classmethod
	def get_assembled(cls):
		return cls._assembled.copy()

	@classmethod
	def is_assembled(cls, state: np.ndarray):
		return (state == cls._assembled).all()
	
	@classmethod
	def as_oh(cls, state: np.ndarray):
		# Takes in a state and returns a 480 long one-hot tensor on the given device
		oh = torch.zeros(480)
		idcs = np.arange(20) * 24 + state
		oh[idcs] = 1
		return oh
	
	@staticmethod
	def as633(state: np.ndarray):
		"""
		Order: F, B, T, D, L, R
		"""
		state633 = (np.ones((3, 3, 6)) * np.arange(6)).transpose(2, 1, 0)
		for i in range(8):
			pos = np.floor(state[i] / 3)
			orientation = state[i] // 3
			# TODO: Dis gonna suck
			# Probably need to map each position to all stickers, so huge map
		for i in range(12):
			pos = np.floor(state[i] / 2)
			orientation = state[i] // 2
		return state633
	
	@staticmethod
	def stringify(state: np.ndarray):
		# TODO: Fancy print
		return str(state)

if __name__ == "__main__":
	
	state = Cube.get_assembled()
	print(state)
	state = Cube.rotate(state, 0, False)
	print(state)
	
	# Benchmarking example
	from src.rubiks.utils.benchmark import Benchmark
	def test_scramble(games):
		# Function is weird, as it is designed to work for both single and multithreaded benchmarks
		if hasattr(games, "__iter__"):
			for _ in games:
				Cube.as_oh(Cube.scramble(n)[0])
		else:
			Cube.as_oh(Cube.scramble(n)[0])

	n = int(1e5)
	nt = range(1, 7)
	games = [None] * 24

	title = f"Scramble bench: {len(games)} cubes each with {n} scrambles"
	bm = Benchmark(test_scramble, "local_benchmarks/scramble_2024", title)
	bm.singlethreaded("Using 20 x 24 representation", games)
	threads, times = bm.multithreaded(nt, games, "Using 20 x 24 representation")
	bm.plot_mt_results(threads, times, f"{title} using 20 x 24")

	
	

