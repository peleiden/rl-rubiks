"""
The API for our Rubik's Cube simulator.
As we use the Rubik's cube in many different ways, this module has some requirements

- Pretty much all functions must work without altering any state as they are to be used by agents.
- Efficiency
- Must support two different Rubik's representations (20x24 and 6x8x6)

The solution to this is this quite large module with these features

- The module carries NO STATE: Environment state must be maintained elsewhere when used and this API works
	mostly purely functionally
- Many functions are polymorphic between corresponding functions of the two representations
- Most functions are implemented compactly using numpy or pytorch sacrificing readability for efficiency
- Some global constants are maintained
"""
import numpy as np
from numpy import arange, arange, repeat
import torch

from librubiks import cpu, gpu, get_is2024, set_is2024
from librubiks.cube.maps import SimpleState, get_corner_pos, get_side_pos, get_tensor_map, get_633maps, neighbors_686


####################
# Action constants #
####################

# If the six sides are represented by an array, the order should be F, B, T, D, L, R
F, B, T, D, L, R = 0, 1, 2, 3, 4, 5
action_names = ('F', 'B', 'T', 'D', 'L', 'R')

action_space = list()
for i in range(6): action_space.extend( [(i, 1), (i, 0)] )
action_dim = len(action_space)

################
# Rotate logic #
################

def rotate(state: np.ndarray, face: int, direction: int) -> np.ndarray:
	"""
	Performs one move on the cube, specified by the side (0-5),
	and if the direction is negative (0) or positive (1)
	"""
	method = _Cube2024.rotate if get_is2024() else _Cube686.rotate
	return method(state, face, direction)

def multi_rotate(states: np.ndarray, faces: np.ndarray, directions: np.ndarray) -> np.ndarray:
	# Performs action (faces[i], directions[i]) on states[i]
	method = _Cube2024.multi_rotate if get_is2024() else _Cube686.multi_rotate
	return method(states, faces, directions)

#################
# Solving logic #
#################

def _get_2024solved(dtype):
	solved_state = SimpleState()
	tensor_state = np.empty(20, dtype=dtype)
	for i in range(8):
		tensor_state[i] = get_corner_pos(solved_state.corners[i], solved_state.corner_orientations[i])
	for i in range(12):
		tensor_state[i+8] = get_side_pos(solved_state.sides[i], solved_state.side_orientations[i])
	return tensor_state

def _get_686solved(dtype):
	solved_state = np.zeros((6, 8, 6), dtype=dtype)
	for i in range(6):
		solved_state[i, :, i] = 1
	return solved_state

dtype = np.int8  # Data type used for internal representation
_solved2024 = _get_2024solved(dtype)
_solved686 = _get_686solved(dtype)

def get_solved_instance() -> np.ndarray:
	# Careful, Ned, careful now - this method returns the instance - not a copy - so the output is readonly
	# If speed is not critical, use get_solved()
	return _solved2024 if get_is2024() else _solved686

def get_solved() -> np.ndarray:
	return get_solved_instance().copy()

def is_solved(state: np.ndarray) -> bool:
	return (state == get_solved_instance()).all()

def multi_is_solved( states: np.ndarray) -> np.ndarray:
	return (states == get_solved_instance()).all(axis=tuple(range(1, len(shape())+1)))

########################
# Representation logic #
########################

def shape():
	return get_solved_instance().shape

def as_oh(states: np.ndarray) -> torch.tensor:
	# Takes in n states and returns an n x 480 one-hot tensor
	method = _Cube2024.as_oh if get_is2024() else _Cube686.as_oh
	return method(states)

def as_correct(t: torch.tensor) -> torch.tensor:
	assert not get_is2024(), "Correctness representation is only implemented for 20x24 representation"
	return _Cube686.as_correct(t)

def get_oh_shape() -> int:
	return 480 if get_is2024() else 288

def repeat_state(state: np.ndarray, n: int=action_dim) -> np.ndarray:
	"""
	Repeats state n times, such that the output array will have shape n x *Cube shape
	Useful in combination with multi_rotate
	"""
	return np.tile(state, [n, *[1]*len(shape())])

def as633(state: np.ndarray) -> np.ndarray:
	"""
	Order: F, B, T, D, L, R
	"""
	method = _Cube2024.as633 if get_is2024() else _Cube686.as633
	return method(state)

def as69(state: np.ndarray) -> np.ndarray:
	# Nice
	return as633(state).reshape((6, 9))

def stringify(state: np.ndarray) -> str:
	state633 = as633(state)
	stringarr = np.empty((9, 12), dtype=str)
	stringarr[...] = " "
	simple = np.array([
		[-1, T, -1, -1],
		[L,  F,  R,  B],
		[-1, D, -1, -1],
	])
	for i in range(6):
		pos = tuple(int(x) for x in np.where(simple==i))
		stringarr[pos[0]*3 : pos[0]*3+3, pos[1]*3 : pos[1]*3+3] = state633[i].astype(str)
	string = "\n".join([" ".join(list(y)) for y in stringarr])
	return string

################
# Action logic #
################

def iter_actions(n: int=1):
	"""
	Returns a numpy array of size 2 x n*action_dim containing tiled actions
	Practical for use with multi_rotate, e.g. multi_rotate(states, *cube.iter_actions())
	"""
	return np.array(list(zip(*action_space*n)), dtype=np.uint8)

def indices_to_actions(indices: np.ndarray) -> (np.ndarray, np.ndarray):
	"""
	Converts an array of action indices [0, 12[ to arrays of corresponding faces and dirs
	"""
	faces = indices // 2
	dirs = ~(indices % 2) + 2
	return faces, dirs

def rev_action(action: int) -> int:
	return action + 1 if action % 2 == 0 else action - 1

def rev_actions(actions: np.ndarray) -> np.ndarray:
	rev_actions = actions - 1
	rev_actions[actions % 2 == 0] += 2
	return rev_actions

##################
# Scramble logic #
##################

def scramble(depth: int, force_not_solved=False) -> (np.ndarray, np.ndarray, np.ndarray):
	faces = np.random.randint(6, size=(depth,))
	dirs = np.random.randint(2, size=(depth,))
	state = get_solved()
	for face, d in zip(faces, dirs):
		state = rotate(state, face, d)

	if force_not_solved and is_solved(state) and depth != 0:
		return scramble(depth, True)

	return state, faces, dirs

def sequence_scrambler(games: int, depth: int, with_solved: bool) -> (np.ndarray, torch.tensor):
	"""
	An out-of-place scrambler which returns the state to each of the scrambles useful for ADI
	Returns a games x n x 20 tensor with states as well as their one-hot representations (games * n) x 480
	:with_solved: Whether to include the solved cube in the sequence
	"""
	states = []
	current_states = np.array([get_solved_instance()]*games)
	faces = np.random.randint(0, 6, (depth, games))
	dirs = np.random.randint(0, 2, (depth, games))
	if with_solved: states.append(current_states)
	for d in range(depth - with_solved):
		current_states = multi_rotate(current_states, faces[d], dirs[d])
		states.append(current_states)
	states = np.vstack(np.transpose(states, (1, 0, *np.arange(2, len(shape())+2))))
	oh_states = as_oh(states)
	return states, oh_states


class _Cube2024:

	maps = get_tensor_map(dtype)
	corner_side_idcs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	corner_633map, side_633map = get_633maps(F, B, T, D, L, R)

	@classmethod
	def rotate(cls, state: np.ndarray, face: int, direction: int):
		"""
		Performs one move on the cube, specified by the side (0-5),
		and whether the rotation is in a positive direction (0 for negative and 1 for positive)
		"""
		
		map_ = cls.maps[direction, face]
		state = state + map_[cls.corner_side_idcs, state]
		
		return state

	@classmethod
	def multi_rotate(cls, states: np.ndarray, faces: np.ndarray, directions: np.ndarray):
		# Performs action (faces[i], directions[i]) on states[i]
		altered_states = states.copy()
		maps = cls.maps[directions, faces]
		idcs8 = repeat(arange(len(states)), 8)
		idcs12 = repeat(arange(len(states)), 12)
		altered_states[:, :8] += maps[idcs8, 0, altered_states[:, :8].ravel()].reshape((-1, 8))
		altered_states[:, 8:] += maps[idcs12, 1, altered_states[:, 8:].ravel()].reshape((-1, 12))
		return altered_states

	@staticmethod
	def as_oh(states: np.ndarray):
		# Takes in n states and returns an n x 480 one-hot tensor
		if len(states.shape) == 1:
			oh = torch.zeros(1, 480, device=gpu)
			idcs = np.arange(20) * 24 + states
			oh[0, idcs] = 1
		else:
			oh = torch.zeros(states.shape[0], 480, device=gpu)
			idcs = np.arange(20) * 24 + states
			all_idcs = np.repeat(np.arange(len(states)), 20)
			oh[all_idcs, idcs.ravel()] = 1
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


class _Cube686:
	
	# No shame
	adjacents = np.array([6, 7, 0, 2, 3, 4, 4, 5, 6, 0, 1, 2])
	rolled_adjecents = np.roll(adjacents, 3)
	n3_03 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
	n3_n13 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2])
	roll_left = np.array([2, 3, 4, 5, 6, 7, 0, 1])
	roll_right = np.array([6, 7, 0, 1, 2, 3, 4, 5])

	# Maps an 8 long vector starting at (0, 0) in 3x3 onto a 9 long vector which can be reshaped to 3x3
	map633 = np.array([0, 3, 6, 7, 8, 5, 2, 1])
	# Number of times the 8 long vector has to be shifted to the left to start at (0, 0) in 3x3
	shifts = np.array([0, 6, 6, 4, 2, 4])

	solved_cuda = torch.from_numpy(_get_686solved(dtype)).to(gpu)

	@classmethod
	def rotate(cls, state: np.ndarray, face: int, direction: int):
		"""
		Performs one move on the cube, specified by the side (0-5),
		and if the direction is negative (0) or positive (1)
		"""

		altered_state = state.copy()
		ini_state = state[neighbors_686[face]]

		if direction:
			altered_state[face] = state[face, cls.roll_right]
			as_idcs0 = neighbors_686[[face]*12, cls.n3_03]
			altered_state[as_idcs0, cls.adjacents] = ini_state[cls.n3_n13, cls.rolled_adjecents]
		else:
			altered_state[face] = state[face, cls.roll_left]
			as_idcs0 = neighbors_686[[face]*12, cls.n3_n13]
			altered_state[as_idcs0, cls.rolled_adjecents] = ini_state[cls.n3_03, cls.adjacents]

		return altered_state

	@staticmethod
	def multi_rotate(states: np.ndarray, faces: np.ndarray, directions: np.ndarray):
		return np.array([rotate(state, face, direction) for state, face, direction in zip(states, faces, directions)], dtype=dtype)

	@staticmethod
	def as_oh(states: np.ndarray) -> torch.tensor:
		# This representation is already one-hot encoded, so only ravelling is done
		if len(states.shape) == 3:
			states = np.expand_dims(states, 0)
		states = torch.from_numpy(states.reshape(len(states), 288)).to(gpu).float()
		return states

	@classmethod
	def as_correct(cls, t: torch.tensor) -> torch.tensor:
		"""
		oh is a one-hot encoded tensor of shape n x 288 as produced by _Cube686.as_oh
		This methods creates a correctness representation of the tensor of shape n x 6 x 8
		"""
		# TODO: Write tests for this method
		oh = t.reshape(len(t), 6, 8, 6).to(gpu)
		correct_repr = torch.all(oh[:] == cls.solved_cuda, dim=3).long()
		correct_repr[correct_repr==0] = -1
		return correct_repr.float()

	@classmethod
	def as633(cls, state: np.ndarray):
		state68 = np.where(state == 1)[2].reshape((6, 8))
		state69 = (np.ones((9, 6)) * np.arange(6)).astype(int).T  # Nice
		for i in range(6):
			state69[i, cls.map633] = np.roll(state68[i], -cls.shifts[i], axis=0)
		return state69.reshape((6, 3, 3))

