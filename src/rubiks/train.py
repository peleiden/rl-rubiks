import matplotlib.pyplot as plt
import numpy as np
import os

from src.rubiks.utils.logger import Logger, NullLogger
from src.rubiks.cube import RubiksCube

import torch

class Train:

	def __init__(self, optim, loss_fn, logger = NullLogger()):
		self.optim = optim
		self.loss_fn = loss_fn
		self.log = logger
		self.log("Created trainer with", f"Optimizer: {self.optim}", f"Loss function: {self.loss_fn}\n")
		
	def train(self, net, rollouts: int, batch_size: int = 50, rollout_games: int = 100, rollout_depth: int = 10,  evaluation_interval: int = 2, learning_rate = 1e-2):
		'''
		Trains `net` for `rollouts` rollouts each consisting of `rollout_games` games and scrambled for `rollout_depth`. 
		'''
		self.log(f"Beginning training")
		self.log(f"Rollouts: {rollouts} each consisting of {rollout_games} games with a depth of {rollout_depth}. Every {evaluation_interval}, the model is evaluated.")
		
		optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
		policy_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
		value_criterion  = torch.nn.MSELoss(reduction = 'none')
		
		self.log(f"Optimizer: {optimizer}, policy and value criterions: {policy_criterion}, {value_criterion}")
		train_losses = np.empty(rollouts)


		eval_rollouts = list()
		eval_rewards = list()
		for rollout in range(rollouts):
			net.train()
			training_data, targets, loss_weights = self.ADI_traindata(net, rollout_games, rollout_depth)
			training_data, loss_weights = torch.Tensor(training_data), torch.Tensor(loss_weights)

			for batch in self._gen_batches_idcs(batch_size):
				policy_pred, value_pred = net(training_data[batch], policy = True, value = True)
				losses = policy_criterion(policy_pred, targets[batch][0])
				losses += value_criterion(value_pred, targets[batch][1])
				losses *= loss_weights[batch]

				loss = losses.mean()
				loss.backward()
				optimizer.step()

				train_losses[rollout] = float(loss)
				torch.cuda.empty_cache()		

			if (rollout + 1) % evaluation_interval == 0:
				net.eval()
				eval_reward = NotImplementedError
				eval_rollouts.append(rollout)
				eval_rewards.append(eval_reward)
		
		return net, train_losses, eval_rollouts, eval_rewards
	
	def ADI_traindata(self, net, games: int, sequence_length: int):
		'''
		Implements Autodidactic iteration as per McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge" section 4.1
		Returns games * sequence_length number of data points in three arrays. 
		np.array: `states` contains the rubiks state for each data point
		np.array: `targets` contains optimal value and policy targets for each training point
		np.array: `loss_weights` contains the weight for each training point (see weighted samples subsection)
		'''
		with torch.no_grad():
			cube = RubiksCube()

			states  		= np.empty((games*sequence_length, *cube.assembled.shape))
			targets			= np.empty((games*sequence_length, 2))
			loss_weights	= np.empty(games*sequence_length)
			
			#Plays a number of games
			for i in range(games):
				scrambled_cubes = cube.sequence_scrambler(sequence_length)
				states[i:i+sequence_length] = scrambled_cubes

				#For all states in the scrambled game
				for j, scrambled_state in enumerate(scrambled_cubes):
					subvalues = np.empty(cube.action_dim)
					
					#Explore 12 substates
					for k, action in enumerate(cube.action_space):
						substate = cube.rotate(scrambled_state, *action)

						value = float(
							net(substate, policy = False, value = True)
						)

						value += 1 if (substate == cube.assembled).all() else -1
						subvalues[k] = value

					policy = subvalues.argmax()

					targets[i+j, 0] 	= policy
					targets[i+j, 1] 	= subvalues[policy]
					loss_weights[i+j]	= 1/(sequence_length-j)
				
		return states, targets, loss_weights 
	
	@staticmethod
	def _gen_batches_idcs(size: int, bsize: int):
		# Genererer batches
		# Batch = antal billeder den træner på før modellen opdateres
		nbatches = size // bsize
		idcs = np.arange(size)
		np.random.shuffle(idcs)
		for batch in range(nbatches):
			yield idcs[batch * bsize:(batch + 1) * bsize]


	@staticmethod
	def plot_training(val_epochs, train_losses, val_losses, save_dir: str, title="", show=False):
		
		plt.figure(figsize=(19.2, 10.8))
		plt.plot(val_epochs, train_losses, "b", label="Training loss")
		plt.plot(val_epochs, val_losses, "r-", label="Validation loss")
		plt.title(title if title else "Training loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		os.makedirs(save_dir, exist_ok=True)
		plt.savefig(os.path.join(save_dir, "training.png"))
		if show:
			plt.show()
