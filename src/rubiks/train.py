import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.rubiks.cube.cube import Cube
from src.rubiks.post_train.agents import DeepCube
from src.rubiks.post_train.evaluation import Evaluator
from src.rubiks.utils.device import cpu, gpu
from src.rubiks.utils.logger import Logger, NullLogger
from src.rubiks.utils.ticktock import TickTock

class Train:
	
	moves_per_rollout: int
	
	train_losses: np.ndarray
	train_rollouts: np.ndarray
	eval_rollouts = list()
	eval_rewards = list()


	def __init__(self,
				 device,
				 optim_fn				= torch.optim.RMSprop,
				 lr: float				= 1e-5,
				 policy_criterion		= torch.nn.CrossEntropyLoss,
				 value_criterion		= torch.nn.MSELoss,
				 logger: Logger			= NullLogger(),
				 eval_scrambling: dict 	= None,
				 eval_max_moves: int	= None,
		):
		
		self.device = device

		self.optim 	= optim_fn
		self.lr		= lr

		self.policy_criterion = policy_criterion(reduction = 'none')
		self.value_criterion  = value_criterion(reduction = 'none')

		self.log = logger
		self.log(f"Created trainer with optimizer: {self.optim}, policy and value criteria: {self.policy_criterion}, {self.value_criterion}. Learning rate: {self.lr}")
		self.tt = TickTock()

		agent = DeepCube(net = None)
		self.evaluator = Evaluator(agent, max_moves = eval_max_moves, scrambling_depths= eval_scrambling, verbose = False, logger = self.log)

	def train(self,
			  net,
			  rollouts: int,
			  batch_size: int			= 50,  # Required to be > 1 when training with batchnorm
			  rollout_games: int		= 10000,
			  rollout_depth: int		= 200,
			  evaluation_interval: int	= 2,
			  evaluation_length: int	= 20,
			  verbose: bool			= True,
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
		self.train_rollouts, self.train_losses = np.arange(rollouts), np.empty(rollouts)
		
		for rollout in range(rollouts):
			torch.cuda.empty_cache()
			
			self.tt.section("Training data")
			training_data, policy_targets, value_targets, loss_weights = self.ADI_traindata(net, rollout_games, rollout_depth)
			self.tt.end_section("Training data")
			self.tt.section("Training data to device")
			training_data, value_targets, policy_targets, loss_weights = training_data.to(self.device),\
																		 torch.from_numpy(value_targets).to(self.device),\
																		 torch.from_numpy(policy_targets).to(self.device),\
																		 torch.from_numpy(loss_weights).to(self.device)
			self.tt.end_section("Training data to device")
			
			self.tt.section("Training loop")
			net.train()
			batch_losses = list()
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
				
				batch_losses.append(float(loss))
			self.train_losses[rollout] = np.mean(batch_losses)
			self.tt.end_section("Training loop")
			
			torch.cuda.empty_cache()
			if verbose or rollout in (np.linspace(0, 1, 20)*rollouts).astype(int):
				self.log(f"Rollout {rollout} completed with mean loss {self.train_losses[rollout]}.")
			
			if evaluation_interval and (rollout + 1) % evaluation_interval == 0:
				self.tt.section("Evaluation")
				net.eval()
				self.evaluator.agent.update_net(net)
				eval_results = self.evaluator.eval(evaluation_length)
				eval_reward = (eval_results != 0).mean()  # TODO: This reward should be smarter than simply counting the frequency of completed games within max_moves :think:
				
				self.eval_rollouts.append(rollout)
				self.eval_rewards.append(eval_reward)
				self.tt.end_section("Evaluation")
		
		if verbose:
			print(self.tt)
		
		return net

	def ADI_traindata(self, net, games: int, sequence_length: int):
		"""
		Implements Autodidactic Iteration as per McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge" section 4.1

		Returns games * sequence_length number of observations divided in four arrays:

		torch.tensor: `states` contains the rubiks state for each data point
		np.arrays: `policy_targets` and `value_targets` contains optimal value and policy targets for each training point
		np.array: `loss_weights` contains the weight for each training point (see weighted samples subsection of McAleer et al paper)
		"""
		with torch.no_grad():
			# TODO Parallize and consider cpu/gpu conversion (probably move net to cpu and parallelize)
			net.eval()

			N_data = games * sequence_length
			states = torch.empty(N_data, 480).float()
			policy_targets, value_targets = np.empty(N_data, dtype=np.int64), np.empty(games * sequence_length, dtype=np.float32)
			loss_weights = np.empty(N_data)

			# Plays a number of games
			for i in range(games):
				# TODO: Generate before loop and parallelize
				scrambled_cubes = Cube.sequence_scrambler(sequence_length)
				states[i*sequence_length:i*sequence_length + sequence_length] = Cube.as_oh(scrambled_cubes)
				
				# For all states in the scrambled game
				for j, scrambled_state in enumerate(scrambled_cubes):

					# Explore 12 substates
					substates = np.empty((Cube.action_dim, *Cube.solved.shape))
					for k, action in enumerate(Cube.action_space):
						substates[k] = Cube.rotate(scrambled_state, *action)
					rewards = torch.Tensor([1 if Cube.is_solved(substate) else -1 for substate in substates])
					substates_oh = Cube.as_oh(substates).to(self.device)
					
					# TODO: See if possible to move this part to after loop to parallellize further on gpu
					self.tt.section("ADI feedforward")
					values = net(substates_oh, policy=False, value=True).squeeze().cpu()
					self.tt.end_section("ADI feedforward")
					values += rewards

					policy = values.argmax()

					current_idx = i * sequence_length + j
					policy_targets[current_idx] = policy
					value_targets[current_idx] = values[policy] if not Cube.is_solved(scrambled_state) else 0  # Max Lapan convergence fix

					loss_weights[current_idx] = 1 / (j+1)  # TODO Is it correct?
		
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
		plt.semilogy()
		plt.grid(True)
		
		os.makedirs(save_dir, exist_ok=True)
		plt.savefig(os.path.join(save_dir, "training.png"))
		
		if show: plt.show()

	@staticmethod
	def _gen_batches_idcs(size: int, bsize: int):
		"""
		Generates indices for batch 
		"""
		nbatches = size // bsize
		idcs = np.arange(size)
		np.random.shuffle(idcs)
		for batch in range(nbatches):
			yield idcs[batch * bsize:(batch + 1) * bsize]


if __name__ == "__main__":
	from src.rubiks.model import Model, ModelConfig
	loc = "local_train"
	train_logger = Logger(f"{loc}/training_loop.log", "Training loop")
	tt = TickTock()

	modelconfig = ModelConfig(
		batchnorm=False,
	)
	model = Model(modelconfig, logger=train_logger).to(gpu)

	train = Train(gpu, logger=train_logger, lr=1e-5)
	tt.tick()
	model = train.train(model, 200, batch_size=40, rollout_games=200, rollout_depth=20, evaluation_interval=False)
	train_logger(f"Total training time: {tt.stringify_time(tt.tock())}")
	model.save(loc)

	train.plot_training(loc, show=False)
