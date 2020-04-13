import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List


from src.rubiks.solving.search import DeepSearcher
from src.rubiks.solving.agents import DeepAgent
from src.rubiks import cpu, gpu, no_grad
from src.rubiks.cube.cube import Cube
from src.rubiks.model import Model, ModelConfig
from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.utils import seedsetter
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
			loss_weighting: str,
			optim_fn,
			lr: float,
			searcher_class: DeepSearcher,
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
		self.loss_weighting = loss_weighting
		assert self.loss_weighting in ["none", "weighted", "adaptive"]
		self.adi_ff_batches = 1  # Number of batches used for feedforward in ADI_traindata. Reduces vram usage

		self.evaluations = np.unique(np.linspace(0, self.rollouts-1, evaluations, dtype=int)) if evaluations else np.array([], dtype=int)
		self.evaluations.sort()

		self.searcher_class = searcher_class

		self.optim = optim_fn
		self.lr	= lr

		self.policy_criterion = policy_criterion(reduction='none')
		self.value_criterion = value_criterion(reduction='none')

		self.log = logger
		self.log(f"Created trainer with optimizer: {self.optim}, policy and value criteria: {self.policy_criterion}, {self.value_criterion}. Learning rate: {self.lr}")
		self.log("\n".join([
			"Training procedure",
			f"Rollouts: {self.rollouts}",
			f"Batch size: {self.batch_size}",
			f"Rollout games: {self.rollout_games}",
			f"Rollout depth: {self.rollout_depth}",
			f"Loss weighting: {self.loss_weighting}",
		]))
		self.tt = TickTock()

		self.evaluator = evaluator

	def train(self, net: Model) -> (Model, Model):
		"""
		Trains `net` for `rollouts` rollouts each consisting of `rollout_games` games and scrambled for `rollout_depth`.
		Every `evaluation_interval` (or never if evaluation_interval = 0), an evaluation is made of the model at the current stage playing `evaluation_length` games according to `self.evaluator`.
		"""
		self.tt.tick()
		self.moves_per_rollout = self.rollout_depth * self.rollout_games
		self.log(f"Beginning training. Optimization is performed in batches of {self.batch_size}")
		self.log("\n".join([
			f"Rollouts: {self.rollouts}",
			f"Each consisting of {self.rollout_games} games with a depth of {self.rollout_depth}",
			f"Evaluations: {len(self.evaluations)}",
		]))
		lowest_loss = float("inf")

		optimizer = self.optim(net.parameters(), lr=self.lr)
		self.train_rollouts, self.train_losses = np.arange(self.rollouts), np.empty(self.rollouts)

		agent = DeepAgent(self.searcher_class(net))

		for rollout in range(self.rollouts):
			# torch.cuda.empty_cache()

			self.tt.section("Training data")
			training_data, policy_targets, value_targets, loss_weights = self.ADI_traindata(net, rollout)
			self.tt.end_section("Training data")
			self.tt.section("Training data to device")
			training_data, value_targets, policy_targets, loss_weights = training_data.to(gpu),\
																		 value_targets.to(gpu),\
																		 policy_targets.to(gpu),\
																		 loss_weights.to(gpu)
			self.tt.end_section("Training data to device")

			self.tt.section("Training loop")
			net.train()
			batches = self._get_batches(self.moves_per_rollout, self.batch_size)
			batch_loss = 0
			for i, batch in enumerate(batches):
				optimizer.zero_grad()
				policy_pred, value_pred = net(training_data[batch], policy = True, value = True)

				# Use loss on both policy and value
				losses = self.policy_criterion(policy_pred, policy_targets[batch])
				losses += self.value_criterion(value_pred.squeeze(), value_targets[batch])

				loss = losses @ loss_weights[batch]  # Weighting of losses according to move importance
				loss.backward()
				optimizer.step()
				batch_loss += loss.cpu().detach()

			self.train_losses[rollout] = batch_loss / len(batches)
			self.tt.end_section("Training loop")
			
			if self.train_losses[rollout] < lowest_loss:
				lowest_loss = self.train_losses[rollout]
				min_net = net.clone()

			# torch.cuda.empty_cache()
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
				
				self.tt.section("Evaluation")
				net.eval()
				agent.update_net(net)
				eval_results = self.evaluator.eval(agent)
				eval_reward = (eval_results != -1).mean()

				self.eval_rewards.append(eval_reward)
				self.tt.end_section("Evaluation")

		self.log.verbose(self.tt)
		self.log(f"Best net found to have loss of {lowest_loss}")
		self.log(f"Total training time: {self.tt.stringify_time(self.tt.tock(), 's')}")

		return net, min_net

	def get_adi_ff_slices(self):
		data_points = self.rollout_games * self.rollout_depth * Cube.action_dim
		slice_size = data_points // self.adi_ff_batches + 1
		# Final slice may have overflow, however this is simply ignored when indexing
		slices = [slice(i*slice_size, (i+1)*slice_size) for i in range(self.adi_ff_batches)]
		return slices

	@no_grad
	def ADI_traindata(self, net, rollout: int):
		"""
		Implements Autodidactic Iteration as per McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge" section 4.1

		Returns games * sequence_length number of observations divided in four arrays:

		torch.tensor: `states` contains the rubiks state for each data point
		np.arrays: `policy_targets` and `value_targets` contains optimal value and policy targets for each training point
		np.array: `loss_weights` contains the weight for each training point (see weighted samples subsection of McAleer et al paper)
		"""

		net.eval()
		self.tt.section("Scrambling")
		states, oh_states = Cube.sequence_scrambler(self.rollout_games, self.rollout_depth)
		self.tt.end_section("Scrambling")

		# Keeps track of solved states - Max Lapan's convergence fix
		solved_scrambled_states = np.array([
			Cube.is_solved(scrambled_state)
			for game_states in states
			for scrambled_state in game_states
		], dtype=bool)
		# Generates possible substates for all scrambled states
		self.tt.section("ADI substates")
		substates = np.array([
			Cube.rotate(scrambled_state, *action)
			for game_states in states
			for scrambled_state in game_states
			for action in Cube.action_space
		], dtype=Cube.dtype)
		substates_oh = Cube.as_oh(substates).to(gpu)
		self.tt.end_section("ADI substates")

		# Generates policy and value targets
		rewards = torch.tensor([1 if Cube.is_solved(substate) else -1 for substate in substates])
		self.tt.section("ADI feedforward")
		while True:
			try:
				value_parts = [net(substates_oh[slice_], policy=False, value=True).squeeze() for slice_ in self.get_adi_ff_slices()]
				values = torch.cat(value_parts).cpu()
				assert values.shape == torch.Size([self.rollout_games*self.rollout_depth*Cube.action_dim])
				break
			except RuntimeError:  # Caused by running out of vram
				self.log.verbose(f"Increasing number of ADI feed forward batches from {self.adi_ff_batches} to {self.adi_ff_batches*2}")
				self.adi_ff_batches *= 2
		self.tt.end_section("ADI feedforward")
		values += rewards
		values = values.reshape(-1, 12)
		policy_targets = torch.argmax(values, dim=1)
		value_targets = values[np.arange(len(values)), policy_targets]
		value_targets[solved_scrambled_states] = 0
		if self.loss_weighting == "adaptive":
			weighted = np.tile(1 / np.arange(1, self.rollout_depth + 1), self.rollout_games)
			unweighted = np.ones_like(weighted)
			alpha = rollout / self.rollouts
			loss_weights = (1-alpha) * weighted + alpha * unweighted
		elif self.loss_weighting == "weighted":
			loss_weights = np.tile(1 / np.arange(1, self.rollout_depth + 1), self.rollout_games)
		else:
			loss_weights = np.ones(self.rollout_games*self.rollout_depth)
		loss_weights /= loss_weights.sum()
		assert np.isclose(loss_weights.sum(), 1)

		return oh_states, policy_targets, value_targets, torch.from_numpy(loss_weights).float()

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

		if len(self.evaluations):
			color = 'blue'
			reward_ax = loss_ax.twinx()
			reward_ax.set_ylabel("Fraction of games won when evaluating", color=color)
			reward_ax.plot(self.evaluations, self.eval_rewards, color=color, label="Evaluation reward")
			reward_ax.tick_params(axis='y', labelcolor=color)

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
	def _get_batches(size: int, bsize: int):
		"""
		Generates indices for batch
		"""
		nbatches = int(np.ceil(size/bsize))
		idcs = np.arange(size)
		np.random.shuffle(idcs)
		batches = [slice(batch*bsize, (batch+1)*bsize) for batch in range(nbatches)]
		batches[-1] = slice(batches[-1].start, size)
		return batches



