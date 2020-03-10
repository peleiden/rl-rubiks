import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.rubiks.cube import RubiksCube
from src.rubiks.data.data_generation import ADI_traindata
from src.rubiks.post_train.agents import DeepCube
from src.rubiks.post_train.evaluation import Evaluator
from src.rubiks.utils.logger import Logger, NullLogger


class Train:
	
	moves_per_rollout: int
	train_losses: np.ndarray
	eval_rollouts: list
	eval_rewards: list

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
			verbose: bool				= True,
		):
		"""
		Trains `net` for `rollouts` rollouts each consisting of `rollout_games` games and scrambled for `rollout_depth`.
		Every `evaluation_interval` (or never if evaluation_interval = 0), an evaluation af the model at the current stage playing `evaluation_length` games according to `self.evaluator`.
		"""
		self.moves_per_rollout = rollout_depth * rollout_games
		self.log(f"Beginning training.")
		self.log(f"Rollouts: {rollouts}. Each consisting of {rollout_games} games with a depth of {rollout_depth}. Eval_interval: {evaluation_interval}.")

		optimizer = self.optim(net.parameters(), lr=self.lr)
		self.train_losses = np.zeros(rollouts)
		
		self.eval_rollouts = list()
		self.eval_rewards = list()
		for rollout in range(rollouts):
			torch.cuda.empty_cache()

			training_data, policy_targets, value_targets, loss_weights = ADI_traindata(net, rollout_games, rollout_depth)
			training_data, value_targets, policy_targets,  loss_weights = torch.from_numpy(training_data), torch.from_numpy(value_targets), torch.from_numpy(policy_targets), torch.from_numpy(loss_weights) #TODO: handle devicing
			
			net.train()
			for batch in self._gen_batches_idcs(rollout_games, batch_size):
				optimizer.zero_grad()

				policy_pred, value_pred = net(training_data[batch], policy = True, value = True)
				#Use loss on both policy and value

				losses = self.policy_criterion(policy_pred, policy_targets[batch]) 
				losses += self.value_criterion(value_pred.squeeze(), value_targets[batch])
					

				#Weighteing of losses according to move importance
				loss = ( losses * loss_weights[batch] ).mean()
				loss.backward()
				optimizer.step()

				self.train_losses[rollout] += float(loss)

			self.train_losses[rollout] /= self.moves_per_rollout
			
			torch.cuda.empty_cache()
			
			self.log(f"Rollout {rollout} completed with loss {self.train_losses[rollout]}.")

			if evaluation_interval and (rollout + 1) % evaluation_interval == 0:
				net.eval()
				self.evaluator.agent.update_net(net)
				eval_results = self.evaluator.eval(evaluation_length)
				eval_reward = (eval_results != 0).mean()  # TODO: This reward should be smarter than simply counting the frequency of completed games within max_moves :think:
				
				self.eval_rollouts.append(rollout)
				self.eval_rewards.append(eval_reward)
		
		return net

	def plot_training(self, save_dir: str, title="", show=False):
		"""
		Visualizes training by showing training loss + evaluation reward in same plot
		"""
		fig, loss_ax = plt.subplots(figsize=(19.2, 10.8)) 
		loss_ax.set_xlabel(f"Rollout of {self.moves_per_rollout} moves")

		color = 'red'
		loss_ax.set_ylabel("Cross Entropy + MSE, weighted", color = color)
		loss_ax.plot(self.train_losses, label="Training loss", color = color)
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
	net = Model(ModelConfig())
	logger = Logger("local_train/training_loop.log", "Training loop")
	train = Train(logger=logger)

	net = train.train(net, 40, batch_size=5, rollout_games=50, rollout_depth=10, evaluation_interval=0)
	train.plot_training("local_tests/local_train", show=True)
