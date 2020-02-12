
import torch
from copy import deepcopy

# The i'th index contain the neighbors of the i'th side in positive direction
neighbors = (
	{  # Front
		1: (slice(None), -1),
		2: (0, slice(None)),
		3: (slice(None), 0),
		4: (-1, slice(None)),
	},
	{  # Left
		0: (slice(None), 0),
		4: (),
		5: (),
		2: (),
	},
	{  # Below
		0: (),
		1: (),
		5: (),
		3: (),
	},
	{  # Right, reverse of left
		2: (),
		5: (),
		4: (),
		0: (),
	},
	{  # Above, reverse of below
		3: (),
		5: (),
		1: (),
		0: (),
	},
	{  # Behind, reverse of front
		4: (),
		3: (),
		2: (),
		1: (),
	},
)


class RubiksCube:

	def __init__(self, n = 3, device = torch.device("cpu")):

		"""
		Shape: 6 x n x n
		Sides:
			0: Front
			1: Left
			2: Below
			3: Right
			4: Above
			5: Behind
		The state is initialized as a solved cube
		Colours are represented as numbers 0 through 5 and initialized such that colour 0 is on side 0 etc.
		
		Indices on a given side:
		Turn a side towards you. The indices for the matrix on that side follow normal matrix indexing
		"""

		if n != 3:
			raise NotImplementedError("Rubik's cube must be 3 x 3, not %i x %i" % (n, n))

		self.n = n
		shape = (6, self.n, self.n)
		self.state = torch.empty(*shape, dtype = torch.uint8, device = device)
		for i in range(6):
			self.state[i] = i
	
	def rotate(self, side: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		if not 0 <= side <= 5:
			raise IndexError("As there are only six sides, the side should be 0-5, not %i" % side)

		# First the corresponding side is rotated
		# np.rot90(a) is recreated as transpose(flip(a))
		# Reversed is flip(transpose(a))
		# https://github.com/pytorch/pytorch/issues/6271

		# Rotates the actual side
		if pos_rev:
			self.state[side] = torch.flip(self.state[side].t())
		else:
			self.state[side] = torch.flip(self.state[side]).t()
		
		# Rotates the parts of the other sides
		# https://pytorch.org/docs/stable/torch.html#torch.roll
		# First makes a tensor referencing all the 4*n values next to the given side that should be shifted
		to_shift = torch.empty(4 * self.n, dtype = torch.uint8)
		for i in range(4):
			to_shift[i*self.n:(i+1)*self.n] = self.state[self.neighbors[side][i], <]
		
	def __str__(self):

		return str(self.state)
	
	def copy(self):

		"""
		Returns a deep copy of self
		"""

		return deepcopy(self)
	
	def _get_slice(self, n: int or None):

		return n if type(n) == int else slice(0, self.n)
	
	def to(self, device: torch.device, in_place = True):

		"""
		Mimicks tensors' to method
		Use only for device changing
		If not in_place, a new RubiksCube instance is returned
		"""

		attrs = ("state", )
		new_rc = self.copy()
		for attr in attrs:
			setattr(
				new_rc,
				attr,
				getattr(self, attr).to(device)
			)
		
		if in_place:
			self = new_rc
		else:
			return new_rc






