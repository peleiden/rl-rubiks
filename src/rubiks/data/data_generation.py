import numpy as np
import torch

from src.rubiks.cube import RubiksCube
from src.rubiks.utils.mp import multi_exec


def ADI_traindata(net, games: int, sequence_length: int):
	"""
	Implements Autodidactic Iteration as per McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge" section 4.1

	Returns games * sequence_length number of observations divided in three arrays:

	np.array: `states` contains the rubiks state for each data point
	np.array: `targets` contains optimal value and policy targets for each training point
	np.array: `loss_weights` contains the weight for each training point (see weighted samples subsection of McAleer et al paper)
	"""
	with torch.no_grad():
		# TODO Parallize and consider cpu/gpu conversion (probably move net to cpu and parrallelize)
		cube = RubiksCube()
		
		states = np.empty((games * sequence_length, *cube.assembled.shape))
		targets = np.empty((games * sequence_length, 2))
		loss_weights = np.empty(games * sequence_length)
		
		# Generates scrambled states
		scrambled_states = multi_exec(cube.sequence_scrambler, games, sequence_length)
		
		# Plays a number of games
		for i in range(games):
			scrambled_cubes = scrambled_states[i]
			states[i:i + sequence_length + 1] = scrambled_cubes
			
			# For all states in the scrambled game
			for j, scrambled_state in enumerate(scrambled_cubes):
				subvalues = np.empty(cube.action_dim)
				
				# Explore 12 substates
				for k, action in enumerate(cube.action_space):
					substate = torch.Tensor(cube.rotate(scrambled_state, *action)).flatten()
					value = float(net(substate, policy=False, value=True))
					value += 1 if (substate == cube.assembled).all() else -1
					subvalues[k] = value
				policy = subvalues.argmax()
				
				targets[i + j, 0] = policy
				targets[i + j, 1] = subvalues[policy]
				loss_weights[i + j] = 1 / (sequence_length - j)
	
	return states, targets, loss_weights