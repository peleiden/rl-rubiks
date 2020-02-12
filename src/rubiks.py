
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
		self.neighbors = (
			(1, 5, 4, 2),  # Front
			(2, 3, 5, 0),  # Left
			(0, 4, 3, 1),  # Top
			(5, 1, 2, 4),  # Back
			(3, 2, 0, 5),  # Right
			(4, 0, 1, 3),  # Bottom
		)
		self.revolution = (
			torch.tensor([6, 7, 0], device = device),
			torch.arange(2, 5, device = device),
			torch.arange(4, 7, device = device),
			torch.arange(0, 3, device = device),
		)
	
	def rotate(self, face: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		if not 0 <= face <= 5:
			raise IndexError("As there are only six sides, the side should be 0-5, not %i" % face)

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
	
	def is_complete(self):

		full_faces = torch.empty(6, dtype = bool)
		for i in range(6):
			full_faces[i] = (self.state[i] == self.state[i, 0]).all()
		
		return full_faces.all()



if __name__ == "__main__":
	rube = RubiksCube()
	print(rube)
	print(rube.is_complete())
	rube.rotate(0, True)
	print(rube)
	print(rube.is_complete())
	rube.rotate(0, False)
	print(rube)
	print(rube.is_complete())

