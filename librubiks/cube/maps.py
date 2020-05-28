import numpy as np
from dataclasses import dataclass

"""
Every cubie has a position 0-7 for corners and 0-11 for sides. A rotation maps every relevant position to another.
Orientations are also mapped. The sticker on the cubie, which is kept track of, has an orientation 0-2 for corners and 0-1 for sides.
The number represents a "priority". The side with the most stickers in the solved configuration has highest priority.
F/B start with 8 each, T/D with 2, and L/R with 0. Due to the symmetry of the cube, there will never be overlaps.
All maps are given in positive revolution.
Seen from the front:
Front layer    Middle layer    Back layer
h0 s0 h3     | s4    s7      | h4 s8  h7
s1    s3     |               | s9     s11
h1 s2 h2     | s5    s6      | h5 s10 h6

Indices in 6x3x3 array
6x3x3 is based on
  T        2
L F R B  4 0 5 1
  D        3
First -> second -> third has a "right turn"
First in each index also sticker value
"""


def get_633maps(F, B, T, D, L, R):
	corner_633map = (
		((F, 0, 0), (L, 0, 2), (T, 2, 0)),
		((F, 2, 0), (D, 0, 0), (L, 2, 2)),
		((F, 2, 2), (R, 2, 0), (D, 0, 2)),
		((F, 0, 2), (T, 2, 2), (R, 0, 0)),
		((B, 0, 2), (T, 0, 0), (L, 0, 0)),
		((B, 2, 2), (L, 2, 0), (D, 2, 0)),
		((B, 2, 0), (D, 2, 2), (R, 2, 2)),
		((B, 0, 0), (R, 0, 2), (T, 0, 2)),
	)
	side_633map = (
		((F, 0, 1), (T, 2, 1)),
		((F, 1, 0), (L, 1, 2)),
		((F, 2, 1), (D, 0, 1)),
		((F, 1, 2), (R, 1, 0)),
		((T, 1, 0), (L, 0, 1)),
		((D, 1, 0), (L, 2, 1)),
		((D, 1, 2), (R, 2, 1)),
		((T, 1, 2), (R, 0, 1)),
		((B, 0, 1), (T, 0, 1)),
		((B, 1, 2), (L, 1, 0)),
		((B, 2, 1), (D, 2, 1)),
		((B, 1, 0), (R, 1, 2)),
	)
	return corner_633map, side_633map


class SimpleState:
	# Used for representation in the readable Actions maps
	# Initialized in solved state
	corners = np.arange(8)
	corner_orientations = np.zeros(8)
	sides = np.arange(12)
	side_orientations = np.zeros(12)
	def __str__(self):
		return f"Corners:             {[int(x) for x in self.corners]}\n" + \
			   f"Corner orientations: {[int(x) for x in self.corner_orientations]}\n" + \
			   f"Sides:               {[int(x) for x in self.sides]}\n" + \
			   f"Side orientations:   {[int(x) for x in self.side_orientations]}"

@dataclass
class ActionMap:
	corner_map: tuple  # Corner mapping in positive revolution
	side_map: tuple  # Side mapping in positive revolution
	corner_static: int  # Corner orientation static - other two switch
	side_switch: bool  # Side orientation switch

class Actions:
	F = ActionMap((0, 1, 2, 3, 0),
				  (0, 1, 2, 3, 0),
				  0,
				  False)
	B = ActionMap((4, 7, 6, 5, 4),
				  (8, 11, 10, 9, 8),
				  0,
				  False)
	T = ActionMap((0, 3, 7, 4, 0),
				  (0, 7, 8, 4, 0),
				  1,
				  True)
	D = ActionMap((1, 5, 6, 2, 1),
				  (2, 5, 10, 6, 2),
				  1,
				  True)
	L = ActionMap((0, 4, 5, 1, 0),
				  (1, 4, 9, 5, 1),
				  2,
				  False)
	R = ActionMap((7, 3, 2, 6, 7),
				  (3, 6, 11, 7, 3),
				  2,
				  False)


def get_corner_pos(pos: int, orientation: int):
	return pos * 3 + orientation

def get_side_pos(pos: int, orientation: int):
	return pos * 2 + orientation

def get_tensor_map(dtype):
	"""
	Returns two maps
	The first is positive revolution, second is negative
	Each is a six long list containg 2x24 mapping tensors
	Order is F, B, T, D, L, R
	Row one for corners [:8] and row two for sides [8:]
	Value at index i in a mapping should be added to i in state representation
	"""
	actions = [Actions.F, Actions.B, Actions.T, Actions.D, Actions.L, Actions.R]
	map_pos = list()
	map_neg = list()
	# Mappings for each action
	for i in range(6):
		action = actions[i]
		pos = np.zeros((2, 24), dtype=dtype)
		neg = np.zeros((2, 24), dtype=dtype)
		# Mappings for each corner/side cubies
		for j in range(4):
			# Mappings for corners
			for k in range(3):
				new_orientation = k if k == action.corner_static else next(iter({0, 1, 2} - {action.corner_static, k}))
				from_idx = get_corner_pos(action.corner_map[j], k)
				to_idx = get_corner_pos(action.corner_map[j+1], new_orientation)
				pos[0, from_idx] = to_idx - from_idx
				neg[0, to_idx] = from_idx - to_idx
			# Mappings for sides
			for k in range(2):
				new_orientation = k if not action.side_switch else int(not k)
				from_idx = get_side_pos(action.side_map[j], k)
				to_idx = get_side_pos(action.side_map[j+1], new_orientation)
				pos[1, from_idx] = to_idx - from_idx
				neg[1, to_idx] = from_idx - to_idx
		map_pos.append(pos)
		map_neg.append(neg)
	
	return map_pos, map_neg

if __name__ == "__main__":
	# Pretty print of tensor maps
	map_pos, map_neg = get_tensor_map(np.int8)
	for pos, neg in zip(map_pos, map_neg):
		print("".join([f"{x: 4}" for x in pos[0]]))
		print("".join([f"{x: 4}" for x in pos[1]]))
		print("".join([f"{x: 4}" for x in neg[0]]))
		print("".join([f"{x: 4}" for x in neg[1]]))
		print()


