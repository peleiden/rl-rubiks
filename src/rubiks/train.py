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
			lr: float				= 1e-2,
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
			batch_size: int				= 50,
			rollout_games: int 			= 100,
			rollout_depth: int 			= 10,
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
		self.log(f"Rollouts: {rollouts} each consisting of {rollout_games} games with a depth of {rollout_depth}. Eval_interval: {evaluation_interval}.")

		optimizer = self.optim(net.parameters(), lr=self.lr)
		self.train_losses = np.empty(rollouts)
		
		self.eval_rollouts = list()
		self.eval_rewards = list()
		for rollout in range(rollouts):
			net.train()

			training_data, targets, loss_weights = ADI_traindata(net, rollout_games, rollout_depth)
			torch.cuda.empty_cache()
			training_data, loss_weights = torch.Tensor(training_data), torch.Tensor(loss_weights) #TODO: handle devicing

			for batch in self._gen_batches_idcs(rollout_games, batch_size):
				policy_pred, value_pred = net(training_data[batch].flatten(), policy = True, value = True)
				#Use loss on both policy and value
				losses = self.policy_criterion(policy_pred, targets[batch][0])
				losses += self.value_criterion(value_pred, targets[batch][1])
				
				#Weighteing of losses according to move importance
				loss = ( losses * loss_weights[batch] ).mean()
				
				loss.backward()
				optimizer.step()

				self.train_losses[rollout] = float(loss)
				torch.cuda.empty_cache()		

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

		color = 'blue'
		reward_ax = loss_ax.twinx()
		reward_ax.set_ylabel("Cross Entropy + MSE, weighted", color=color)
		reward_ax.plot(self.eval_rollouts, self.eval_rewards, color=color,  label="Evaluation reward")
		reward_ax.tick_params(axis='y', labelcolor=color)


		fig.tight_layout()
		fig.title(title if title else "Training")
		
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
	logger = Logger("local_tests/local_train_test", "Training loop")
	train = Train(logger=logger)

	net = train.train(net, 2, batch_size=20, rollout_games=2, rollout_depth=5, evaluation_interval=0)
	train.plot_training("local_tests/local_train", show=True)
