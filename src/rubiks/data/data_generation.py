import numpy as np
import torch

from src.rubiks.cube import RubiksCube
from src.rubiks.utils.mp import multi_exec


def ADI_traindata(net, games: int, sequence_length: int):
	"""
	Implements Autodidactic Iteration as per McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge" section 4.1

	Returns games * sequence_length number of observations divided in three arrays:

	np.array: `states` contains the rubiks state for each data point
	np.arrays: `policy_targets` and `value_targets` contains optimal value and policy targets for each training point
	np.array: `loss_weights` contains the weight for each training point (see weighted samples subsection of McAleer et al paper)
	"""
	with torch.no_grad():
		# TODO Parallize and consider cpu/gpu conversion (probably move net to cpu and parrallelize)
		net.eval() 
		cube = RubiksCube()

		N_data = games * sequence_length
		states = np.empty(( N_data,  *cube.assembled.shape ), dtype=np.float32)
		policy_targets, value_targets = np.empty(N_data, dtype=np.int64), np.empty(games * sequence_length, dtype=np.float32)
		loss_weights = np.empty(N_data)

		# Plays a number of games
		for i in range(games):
			scrambled_cubes = cube.sequence_scrambler(sequence_length)
			states[i:i + sequence_length] = scrambled_cubes
			
			# For all states in the scrambled game
			for j, scrambled_state in enumerate(scrambled_cubes):

				# Explore 12 substates
				substates = np.empty( (cube.action_dim, *cube.assembled.shape) )
				for k, action in enumerate(cube.action_space): 
					substates[k] = cube.rotate(scrambled_state, *action)
				
				rewards = torch.Tensor( [ 1 if (substate == cube.assembled).all() else -1 for substate in substates ] ) 
				substates = torch.Tensor( substates.reshape(cube.action_dim, -1) ) #TODO: Handle device

				values = net(substates, policy=False, value=True).squeeze()
				values += rewards				

				policy = values.argmax()

				current_idx = i*sequence_length + j
				policy_targets[current_idx] = policy
				value_targets[current_idx] = values[policy]


				loss_weights[current_idx] = 1 / (j+1)  # TODO Is it correct?
	
	states = states.reshape(N_data, -1)
	return states, policy_targets, value_targets, loss_weights