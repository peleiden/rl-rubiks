import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.rubiks.cube import RubiksCube
from src.rubiks.post_train.agents import DeepCube
from src.rubiks.post_train.evaluation import Evaluator
from src.rubiks.utils.logger import Logger, NullLogger


class Train:
	
	moves_per_rollout: int
	
	train_losses = list()
	train_rollouts = list()
	eval_rollouts = list()
	eval_rewards = list()


	def __init__(self, 
			optim_fn				= torch.optim.RMSprop,
			lr: float				= 1e-5,
			policy_criterion		= torch.nn.CrossEntropyLoss,
			value_criterion			= torch.nn.MSELoss,
			logger: Logger			= NullLogger(),
			eval_scrambling: dict 	= None,
			eval_max_moves: int		= None,
		):

		self.optim 	= optim_fn
		self.lr		= lr

		self.policy_criterion = policy_criterion(reduction = 'none')
		self.value_criterion  = value_criterion(reduction = 'none')


		self.log = logger
		self.log(f"Created trainer with optimizer: {self.optim}, policy and value criteria: {self.policy_criterion}, {self.value_criterion}. Learning rate: {self.lr}")

		agent = DeepCube(net = None)
		self.evaluator = Evaluator(agent, max_moves = eval_max_moves, scrambling_procedure = eval_scrambling, verbose = False, logger = self.log)

	def train(self,
			net,
	 		rollouts: int,
			batch_size: int				= 50, #Required to be > 1 when training with batchnorm
			rollout_games: int 			= 10000,
			rollout_depth: int 			= 200,
			evaluation_interval: int 	= 2,
			evaluation_length: int		= 20,
#			verbose: bool				= True,
		):
		"""
		Trains `net` for `rollouts` rollouts each consisting of `rollout_games` games and scrambled for `rollout_depth`.
		Every `evaluation_interval` (or never if evaluation_interval = 0), an evaluation is made of the model at the current stage playing `evaluation_length` games according to `self.evaluator`.
		"""
		self.moves_per_rollout = rollout_depth * rollout_games
		batch_size = self.moves_per_rollout if not batch_size else batch_size 
		self.log(f"Beginning training. Optimization is performed in batches of {batch_size}")
		self.log(f"Rollouts: {rollouts}. Each consisting of {rollout_games} games with a depth of {rollout_depth}. Eval_interval: {evaluation_interval}.")

		optimizer = self.optim(net.parameters(), lr=self.lr)
		
		for rollout in range(rollouts):
			torch.cuda.empty_cache()

			training_data, policy_targets, value_targets, loss_weights = self.ADI_traindata(net, rollout_games, rollout_depth)
			training_data, value_targets, policy_targets,  loss_weights = torch.from_numpy(training_data), torch.from_numpy(value_targets), torch.from_numpy(policy_targets), torch.from_numpy(loss_weights) #TODO: handle devicing
			net.train()
			for batch in self._gen_batches_idcs(self.moves_per_rollout, batch_size):
				optimizer.zero_grad()

				policy_pred, value_pred = net(training_data[batch], policy = True, value = True)
				#Use loss on both policy and value

				losses = self.policy_criterion(policy_pred, policy_targets[batch]) 
				losses += self.value_criterion(value_pred.squeeze(), value_targets[batch])
					
				#Weighteing of losses according to move importance
				loss = ( losses * loss_weights[batch] ).mean()
				loss.backward()
				optimizer.step()

				self.train_losses.append(float(loss)); self.train_rollouts.append(rollout)
			
			torch.cuda.empty_cache()
			self.log(f"Rollout {rollout} completed with mean loss {np.mean(self.train_losses[-(self.moves_per_rollout // batch_size):])}.")

			if evaluation_interval and (rollout + 1) % evaluation_interval == 0:
				net.eval()
				self.evaluator.agent.update_net(net)
				eval_results = self.evaluator.eval(evaluation_length)
				eval_reward = (eval_results != 0).mean()  # TODO: This reward should be smarter than simply counting the frequency of completed games within max_moves :think:
				
				self.eval_rollouts.append(rollout)
				self.eval_rewards.append(eval_reward)
		
		return net


	@staticmethod 
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
				states[i*sequence_length:i*sequence_length + sequence_length] = scrambled_cubes
				
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
					value_targets[current_idx] = values[policy] if not (scrambled_state == cube.assembled).all() else 0  #Max Lapan convergence fix


					loss_weights[current_idx] = 1 / (j+1)  # TODO Is it correct?

		states = states.reshape(N_data, -1)
		return states, policy_targets, value_targets, loss_weights

	def plot_training(self, save_dir: str, title="", show=False):
		"""
		Visualizes training by showing training loss + evaluation reward in same plot
		"""
		fig, loss_ax = plt.subplots(figsize=(19.2, 10.8)) 
		loss_ax.set_xlabel(f"Rollout of {self.moves_per_rollout} moves")

		color = 'red'
		loss_ax.set_ylabel("Cross Entropy + MSE, weighted", color = color)
		loss_ax.plot(self.train_rollouts, self.train_losses, label="Training loss", color = color)
		loss_ax.tick_params(axis='y', labelcolor = color)

		if self.eval_rollouts:
			color = 'blue'
			reward_ax = loss_ax.twinx()
			reward_ax.set_ylabel("Number of games won", color=color)
			reward_ax.plot(self.eval_rollouts, self.eval_rewards, color=color,  label="Evaluation reward")
			reward_ax.tick_params(axis='y', labelcolor=color)


		fig.tight_layout()
		plt.title(title if title else "Training")
		
		os.makedirs(save_dir, exist_ok=True)
		plt.savefig(os.path.join(save_dir, "training.png"))
		
		if show: plt.show()

	@staticmethod
	def _gen_batches_idcs(size: int, bsize: int):
		'''
		Generates indices for batch 
		'''
		nbatches = size // bsize
		idcs = np.arange(size)
		np.random.shuffle(idcs)
		for batch in range(nbatches):
			yield idcs[batch * bsize:(batch + 1) * bsize]


if __name__ == "__main__":
	from src.rubiks.model import Model, ModelConfig
	train_logger = Logger("local_train/training_loop.log", "Training loop")

	modelconfig = ModelConfig(
		batchnorm=False,
	)
	model = Model(modelconfig, logger=train_logger)

	train = Train(logger=train_logger, lr=1e-5)
	model = train.train(model, 40, batch_size=50, rollout_games=100, rollout_depth=20, evaluation_interval=False)

	train.plot_training("local_tests/local_train", show=True)
