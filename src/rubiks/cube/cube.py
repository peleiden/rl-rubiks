import numpy as np
import torch
from src.rubiks.cube.tensor_maps import SimpleState, get_corner_pos, get_side_pos, get_tensor_map, get_633maps

def _get_assembled(dtype):
	assembled_state = SimpleState()
	tensor_state = np.empty(20, dtype=dtype)
	for i in range(8):
		tensor_state[i] = get_corner_pos(assembled_state.corners[i], assembled_state.corner_orientations[i])
	for i in range(12):
		tensor_state[i+8] = get_side_pos(assembled_state.sides[i], assembled_state.side_orientations[i])
	return tensor_state

class Cube:
	
	# If the six sides are represented by an array, the order should be F, B, T, D, L, R
	# For niceness
	F, B, T, D, L, R = 0, 1, 2, 3, 4, 5
	# Corresponding colours
	colours = ["red", "orange", "white", "yellow", "green", "blue"]

	dtype = np.int8  # Data type used for internal representation
	assembled = _get_assembled(dtype)  # READ ONLY!!! Use Cube.get_assembled() if speed is not critical
	map_pos, map_neg = get_tensor_map()
	corner_633map, side_633map = get_633maps(F, B, T, D, L, R)

	action_space = list()
	for i in range(6): action_space.extend( [(i, True), (i, False)] )
	action_dim = len(action_space)
	
	# Scrambling procedure saved as dict for reproducability
	scrambling_procedure = {
		'N_scrambles':	(5, 10),  # Tuple for scrambling random # moves in uniform [low, high[
	}
	
	@classmethod
	def rotate(cls, current_state: np.ndarray, face: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		altered_state = current_state.copy()
		map_ = cls.map_pos[face] if pos_rev else cls.map_neg[face]
		altered_state[:8] += map_[0, altered_state[:8]]
		altered_state[8:] += map_[1, altered_state[8:]]
		
		return altered_state

	@classmethod
	def scramble(cls, n: int):

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)
		state = cls.assembled.copy()

		for face, d in zip(faces, dirs):
			state = cls.rotate(state, face, d)  # Uses rotate instead of move as checking for victory is not needed here
		
		return state, faces, dirs
	
	@classmethod
	def sequence_scrambler(cls, n: int):
		"""
		A non-inplace scrambler which returns the state to each of the scrambles useful for ADI
		"""
		scrambled_states = np.empty((n, *cls.assembled.shape), dtype=cls.dtype)

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)

		scrambled_states[0] = cls.assembled
		for i, face, d in zip(range(n-1), faces, dirs):
			scrambled_states[i+1] = cls.rotate(scrambled_states[i], face, d)
		return scrambled_states
	
	@classmethod
	def get_assembled(cls):
		return cls.assembled.copy()

	@classmethod
	def is_assembled(cls, state: np.ndarray):
		return (state == cls.assembled).all()
	
	@classmethod
	def as_oh(cls, states: np.ndarray):
		# Takes in n states and returns an n x 480 one-hot tensor
		if len(states.shape) == 1:
			states = np.array([states])
		oh = torch.zeros(states.shape[0], 480).squeeze()
		idcs = np.array([np.arange(20) * 24 + state for state in states]).squeeze()
		for i in range(len(idcs)):
			oh[i, idcs[i]] = 1
		return oh
	
	@classmethod
	def as633(cls, state: np.ndarray):
		"""
		Order: F, B, T, D, L, R
		"""
		# Starts with assembled state
		state633 = (np.ones((3, 3, 6)) * np.arange(6)).transpose(2, 1, 0).astype(int)
		for i in range(8):
			# Inserts values for corner i in position pos
			pos = state[i] // 3
			orientation = state[i] % 3
			values = np.roll([x[0] for x in cls.corner_633map[i]], orientation)
			state633[cls.corner_633map[pos][0]] = values[0]
			state633[cls.corner_633map[pos][1]] = values[1]
			state633[cls.corner_633map[pos][2]] = values[2]
		for i in range(12):
			# Inserts values for side i in position pos
			pos = state[i+8] // 2
			orientation = state[i+8] % 2
			values = np.roll([x[0] for x in cls.side_633map[i]], orientation)
			state633[cls.side_633map[pos][0]] = values[0]
			state633[cls.side_633map[pos][1]] = values[1]
		return state633
	
	@classmethod
	def stringify(cls, state: np.ndarray):
		state633 = cls.as633(state)
		stringarr = np.empty((9, 12), dtype=str)
		stringarr[...] = " "
		simple = np.array([
			[-1, cls.T, -1, -1],
			[cls.L, cls.F, cls.R, cls.B],
			[-1, cls.D, -1, -1],
		])
		for i in range(6):
			pos = tuple(int(x) for x in np.where(simple==i))
			stringarr[pos[0]*3 : pos[0]*3+3, pos[1]*3 : pos[1]*3+3] = state633[i].astype(str)
		string = "\n".join([" ".join([x for x in y]) for y in stringarr])
		return string

if __name__ == "__main__":
	
	state = Cube.get_assembled()
	# print(Cube.as633(state))
	print(Cube.stringify(state))
	print()
	state = Cube.rotate(state, 0, True)
	# print(Cube.as633(state))
	print(Cube.stringify(state))
	print()
	# state = Cube.rotate(state, 0, False)
	# print(Cube.as633(state))
	# print()
	
	
	# Benchmarking example
	# from src.rubiks.utils.benchmark import Benchmark
	# def test_scramble(games):
	# 	# Function is weird, as it is designed to work for both single and multithreaded benchmarks
	# 	if hasattr(games, "__iter__"):
	# 		for _ in games:
	# 			Cube.as_oh(Cube.scramble(n)[0])
	# 	else:
	# 		Cube.as_oh(Cube.scramble(n)[0])
	#
	# n = int(1e5)
	# nt = range(1, 7)
	# games = [None] * 24
	#
	# title = f"Scramble bench: {len(games)} cubes each with {n} scrambles"
	# bm = Benchmark(test_scramble, "local_benchmarks/scramble_2024", title)
	# bm.singlethreaded("Using 20 x 24 representation", games)
	# threads, times = bm.multithreaded(nt, games, "Using 20 x 24 representation")
	# bm.plot_mt_results(threads, times, f"{title} using 20 x 24")

	
	

