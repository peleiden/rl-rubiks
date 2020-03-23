from os import cpu_count

import numpy as np
import torch
import torch.multiprocessing as mp
from src.rubiks.cube.maps import SimpleState, get_corner_pos, get_side_pos, get_tensor_map, get_633maps

def _get_solved(dtype):
	assembled_state = SimpleState()
	tensor_state = np.empty(20, dtype=dtype)
	for i in range(8):
		tensor_state[i] = get_corner_pos(assembled_state.corners[i], assembled_state.corner_orientations[i])
	for i in range(12):
		tensor_state[i+8] = get_side_pos(assembled_state.sides[i], assembled_state.side_orientations[i])
	return tensor_state

# Implemented toplevel for multithreading
def _sequence_scrambler(n: int):
	"""
	A non-inplace scrambler that returns the state to each of the scrambles useful for ADI
	"""
	scrambled_states = np.empty((n, *Cube.solved.shape), dtype=Cube.dtype)
	scrambled_states[0] = Cube.get_solved()
	
	faces = np.random.randint(6, size = (n, ))
	dirs = np.random.randint(2, size = (n, )).astype(bool)
	for i, face, d in zip(range(n-1), faces, dirs):
		scrambled_states[i+1] = Cube.rotate(scrambled_states[i], face, d)
	return scrambled_states, Cube.as_oh(scrambled_states)

class Cube:
	
	# If the six sides are represented by an array, the order should be F, B, T, D, L, R
	# For niceness
	F, B, T, D, L, R = 0, 1, 2, 3, 4, 5
	# Corresponding colours
	colours = ["red", "orange", "white", "yellow", "green", "blue"]
	rgba = [
		(1, 0, 0, 1),
		(1, 3/5, 0, 1),
		(1, 1, 1, 1),
		(1, 1, 0, 1),
		(0, 1, 0, 1),
		(0, 0, 1, 1),
	]

	dtype = np.int8  # Data type used for internal representation
	solved = _get_solved(dtype)  # Readonly. Use Cube.get_assembled() if speed is not critical
	map_pos, map_neg = get_tensor_map(dtype)
	corner_633map, side_633map = get_633maps(F, B, T, D, L, R)

	action_space = list()
	for i in range(6): action_space.extend( [(i, True), (i, False)] )
	action_dim = len(action_space)
	
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
		state = cls.solved.copy()
		for face, d in zip(faces, dirs):
			state = cls.rotate(state, face, d)  # Uses rotate instead of move as checking for victory is not needed here
		
		return state, faces, dirs
	
	@classmethod
	def sequence_scrambler(cls, games: int, n: int):
		"""
		A non-inplace scrambler which returns the state to each of the scrambles useful for ADI
		Returns a games x n x 20 tensor with states as well as their one-hot representations (games * n) x 480
		"""
		with mp.Pool(cpu_count()) as p:
			res = p.map(_sequence_scrambler, [n]*games)
			states = np.array([x[0] for x in res])
			oh_states = torch.stack([x[1] for x in res]).view(-1, 480)  # TODO: Vær sikker på, at det er implementeret rigtigt
		return states, oh_states
	
	@classmethod
	def get_solved(cls):
		return cls.solved.copy()

	@classmethod
	def is_solved(cls, state: np.ndarray):
		return (state == cls.solved).all()
	
	@classmethod
	def as_oh(cls, states: np.ndarray):
		# Takes in n states and returns an n x 480 one-hot tensor
		if len(states.shape) == 1:
			states = np.expand_dims(states, 0)
		oh = torch.zeros(states.shape[0], 480)
		idcs = np.array([np.arange(20) * 24 + state for state in states])
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
			# Mapping should probably be smarter
			# For these corners, "right turn" order is 0 2 1 instead of 0 1 2, so orientation is messed up without this fix
			if pos in [0, 2, 5, 7]:
				orientation *= -1
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
	
	state = Cube.get_solved()
	# print(Cube.as633(state))
	# print(Cube.stringify(state))
	# print()
	state = Cube.rotate(state, 2, True)
	Cube.as633(state)
	print(Cube.stringify(state))
	# print()
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

	
	

