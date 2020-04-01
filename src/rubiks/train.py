import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List

import src.rubiks.solving.agents as agents
from src.rubiks.cube.cube import Cube
from src.rubiks.model import Model, ModelConfig
from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.utils import gpu, seedsetter
from src.rubiks.utils.logger import Logger, NullLogger
from src.rubiks.utils.ticktock import TickTock


class Train:

	moves_per_rollout: int

	train_losses: np.ndarray
	train_rollouts: np.ndarray
	eval_rewards = list()
	depths: np.ndarray
	avg_value_targets: List[np.ndarray] = list()

	def __init__(self,
			rollouts: int,
			batch_size: int, # Required to be > 1 when training with batchnorm
			rollout_games: int,
			rollout_depth: int,
			optim_fn,
			lr: float,
			agent: agents.DeepAgent,
			evaluator: Evaluator,
			evaluations: int,
			logger: Logger			= NullLogger(),
			policy_criterion		= torch.nn.CrossEntropyLoss,
			value_criterion			= torch.nn.MSELoss,
		):

		self.rollouts = rollouts
		self.batch_size = self.moves_per_rollout if not batch_size else batch_size
		self.rollout_games = rollout_games
		self.rollout_depth = rollout_depth
		self.depths = np.arange(1, rollout_depth)

		self.evaluations = np.unique(np.linspace(0, self.rollouts, evaluations, dtype=int)) if evaluations else np.array([], dtype=int)
		self.evaluations.sort()

		self.agent_class = agent

		self.optim = optim_fn
		self.lr	= lr

		self.policy_criterion = policy_criterion(reduction='none')
		self.value_criterion = value_criterion(reduction='none')

		self.log = logger
		self.log(f"Created trainer with optimizer: {self.optim}, policy and value criteria: {self.policy_criterion}, {self.value_criterion}. Learning rate: {self.lr}")
		self.log(f"Training procedure:\nRollouts: {self.rollouts}\nBatch size: {self.batch_size}\nRollout games: {self.rollout_games}\nRollout depth: {self.rollout_depth}")
		self.tt = TickTock()

		self.evaluator = evaluator

	def train(self, net: Model) -> Model:
		"""
		Trains `net` for `rollouts` rollouts each consisting of `rollout_games` games and scrambled for `rollout_depth`.
		Every `evaluation_interval` (or never if evaluation_interval = 0), an evaluation is made of the model at the current stage playing `evaluation_length` games according to `self.evaluator`.
		"""
		self.tt.tick()

		#SET SEED
		seedsetter()
		if issubclass(self.agent_class, agents.TreeAgent):
			agent = self.agent_class(net, self.evaluator.max_time)
		else:
			agent = self.agent_class(net)  # FIXME

		self.moves_per_rollout = self.rollout_depth * self.rollout_games
		self.log(f"Beginning training. Optimization is performed in batches of {self.batch_size}")
		self.log("\n".join([
			f"Rollouts: {self.rollouts}",
			f"Each consisting of {self.rollout_games} games with a depth of {self.rollout_depth}",
			f"Evaluations: {len(self.evaluations)}",
		]))

		optimizer = self.optim(net.parameters(), lr=self.lr)
		self.train_rollouts, self.train_losses = np.arange(self.rollouts), np.empty(self.rollouts)

		for rollout in range(self.rollouts):
			torch.cuda.empty_cache()

			self.tt.section("Training data")
			training_data, policy_targets, value_targets, loss_weights = self.ADI_traindata(net, self.rollout_games, self.rollout_depth)
			self.tt.end_section("Training data")
			self.tt.section("Training data to device")
			training_data, value_targets, policy_targets, loss_weights = training_data.to(gpu),\
																		 torch.from_numpy(value_targets).to(gpu),\
																		 torch.from_numpy(policy_targets).to(gpu),\
																		 torch.from_numpy(loss_weights).to(gpu)
			self.tt.end_section("Training data to device")

			self.tt.section("Training loop")
			net.train()
			batch_losses = list()
			for batch in self._gen_batches_idcs(self.moves_per_rollout, self.batch_size):
				optimizer.zero_grad()

				# print(training_data[batch].shape)
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
			if self.log.is_verbose() or rollout in (np.linspace(0, 1, 20)*self.rollouts).astype(int):
				self.log(f"Rollout {rollout} completed with mean loss {self.train_losses[rollout]}")

			if rollout in self.evaluations:
				self.tt.section("Target value average")
				targets = value_targets.cpu().numpy()
				self.avg_value_targets.append(np.empty_like(self.depths, dtype=float))
				for i, depth in enumerate(self.depths):
					idcs = np.arange(self.rollout_games) * self.rollout_depth + depth
					self.avg_value_targets[-1][i] = targets[idcs].mean()
				self.tt.end_section("Target value average")
				# FIXME
				self.tt.section("Evaluation")
				net.eval()
				agent.update_net(net)
				eval_results = self.evaluator.eval(agent)
				eval_reward = (eval_results != 0).mean()

				self.eval_rewards.append(eval_reward)
				self.tt.end_section("Evaluation")

		self.log.verbose(self.tt)
		self.log(f"Total training time: {self.tt.stringify_time(self.tt.tock(), 's')}")

		return net

	def ADI_traindata(self, net, games: int, sequence_length: int):
		"""
		Implements Autodidactic Iteration as per McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge" section 4.1

		Returns games * sequence_length number of observations divided in four arrays:

		torch.tensor: `states` contains the rubiks state for each data point
		np.arrays: `policy_targets` and `value_targets` contains optimal value and policy targets for each training point
		np.array: `loss_weights` contains the weight for each training point (see weighted samples subsection of McAleer et al paper)
		TODO: Figure out why mp doesn't work
		"""

		N_data = games * sequence_length
		self.tt.section("Scrambling")
		states, oh_states = Cube.sequence_scrambler(games, sequence_length)
		self.tt.end_section("Scrambling")
		policy_targets = np.empty(N_data, dtype=np.int64)
		value_targets = np.empty(games * sequence_length, dtype=np.float32)
		loss_weights = np.empty(N_data)

		net.eval()
		prev_grad = torch.is_grad_enabled()
		torch.set_grad_enabled(False)
		# Plays a number of games
		for i in range(games):
			# For all states in the scrambled game
			for j, scrambled_state in enumerate(states[i]):
				# Explore 12 substates
				substates = np.empty((Cube.action_dim, *Cube.get_solved_instance().shape))
				for k, action in enumerate(Cube.action_space):
					substates[k] = Cube.rotate(scrambled_state, *action)
				rewards = torch.Tensor([1 if Cube.is_solved(substate) else -1 for substate in substates])
				substates_oh = Cube.as_oh(substates).to(gpu)

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
		torch.set_grad_enabled(prev_grad)

		return oh_states, policy_targets, value_targets, loss_weights

	def plot_training(self, save_dir: str, title="", semi_logy=False, show=False):
		"""
		Visualizes training by showing training loss + evaluation reward in same plot
		"""
		self.log("Making plot of training")
		fig, loss_ax = plt.subplots(figsize=(19.2, 10.8))
		loss_ax.set_xlabel(f"Rollout of {self.moves_per_rollout} moves")

		color = 'red'
		loss_ax.set_ylabel("Cross Entropy + MSE, weighted", color = color)
		loss_ax.plot(self.train_rollouts, self.train_losses, label="Training loss", color = color)
		loss_ax.tick_params(axis='y', labelcolor = color)

		# if self.eval_rollouts: TODO
		# 	color = 'blue'
		# 	reward_ax = loss_ax.twinx()
		# 	reward_ax.set_ylabel("Number of games won", color=color)
		# 	reward_ax.plot(self.eval_rollouts, self.eval_rewards, color=color, label="Evaluation reward")
		# 	reward_ax.tick_params(axis='y', labelcolor=color)

		fig.tight_layout()
		plt.title(title if title else "Training")
		if semi_logy: plt.semilogy()
		plt.grid(True)

		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "training.png")
		plt.savefig(path)
		self.log(f"Saved plot to {path}")

		if show: plt.show()
		plt.clf()

	def plot_value_targets(self, loc, show=False):
		self.log("Plotting average value targets")
		plt.figure(figsize=(19.2, 10.8))
		for target, rollout in zip(self.avg_value_targets, self.evaluations):
			plt.plot(self.depths, target, label=f"Rollout {rollout}")
		plt.legend(loc=1)
		plt.xlabel("Scrambling depth")
		plt.ylabel("Average target value")
		path = f"{loc}/avg_target_values.png"
		plt.savefig(path)
		if show: plt.show()
		plt.clf()
		self.log(f"Saved plot to {path}")

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



