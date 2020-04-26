import sys, os
from shutil import rmtree

from ast import literal_eval

import numpy as np
import torch

from src.rubiks.utils import seedsetter, get_commit
from src.rubiks.utils.parse import Parser
from src.rubiks.utils.ticktock import get_timestamp
from src.rubiks.utils.logger import Logger

from src.rubiks import cpu, gpu, get_is2024, set_is2024, store_repr, restore_repr
from src.rubiks.model import Model, ModelConfig
from src.rubiks.train import Train

from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.solving.agents import DeepAgent
from src.rubiks.solving.search import PolicySearch

options = {
	'location': {
		'default':  'data/local_train'+get_timestamp(for_file=True),
		'help':	    "Save location for logs and plots",
		'type':	    str,
	},
	'rollouts': {
		'default':  500,
		'help':	    'Number of passes of ADI+parameter update',
		'type':	    int,
	},
	'rollout_games': {
		'default':  100,
		'help':	    'Number of games in ADI in each rollout',
		'type':	    int,
	},
	'rollout_depth': {
		'default':  50,
		'help':	    'Number of scramblings applied to each game in ADI',
		"type":	    int,
	},
	'loss_weighting': {
		'default':  'weighted',
		'help':	    'Weighting method applied to scrambling depths',
		'type':	    str,
		'choices':  ['weighted', 'none', 'adaptive'],
	},
	'batch_size': {
		'default':  50,
		'help':	    'Number of training examples to be used in each parameter update',
		'type':	    int
	},
	'lr': {
		'default':  1e-5,
		'help':	    'Learning rate of parameter update',
		'type':	    float,
	},
	'gamma': {
		'default':  1,
		'help':	    'Learning rate reduction parameter. Learning rate is set updated as lr <- gamma * lr 100 times during training',
		'type':	    float,
	},
	'optim_fn': {
		'default':  'RMSprop',
		'help':	    'Name of optimization function corresponding to class in torch.optim',
		'type':	    str,
	},
	'evaluations': {
		'default':  200,
		'help':	    'Number of evaluations during training',
		'type':	    int,
	},
	'is2024': {
		'default':  True,
		'help':	    'True for 20x24 Rubiks representation and False for 6x8x6',
		'type':	    literal_eval,
		'choices':  [True, False],
	},
	'arch': {
		'default':	'fc',
		'help':		'Network architecture. fc for fully connected, res for fully connected with residual blocks, and conv for convolutional blocks',
		'type':		str,
		'choices':	['fc', 'res', 'conv'],
	},
}



class TrainJob:
	def __init__(self,
			name: str,
			# Set by parser, should correspond to values in `options`  above and defaults can be controlled there
			location: str,
			rollouts: int,
			rollout_games: int,
			rollout_depth: int,
			loss_weighting: str,
			batch_size: int,
			lr: float,
			gamma: float,
			optim_fn: str,
			evaluations: int,
			is2024: bool,
			arch: str,

			# Currently not set by argparser/configparser

			agent = DeepAgent(PolicySearch(None, True)),
			eval_games: int = 200,
			max_time: float = .01,
			scrambling_depths: tuple = (8,),

			verbose: bool = True,
		):

		self.name = name
		assert isinstance(self.name, str)

		self.rollouts = rollouts
		assert self.rollouts > 0
		self.rollout_games = rollout_games
		assert self.rollout_games > 0
		self.rollout_depth = rollout_depth
		assert rollout_depth > 0

		self.loss_weighting = loss_weighting
		assert loss_weighting in ["adaptive", "weighted", "none"]
		self.batch_size = batch_size
		assert 0 < self.batch_size <= self.rollout_games * self.rollout_depth
		self.lr = lr
		assert float(lr) and lr <= 1
		self.gamma = gamma
		assert 0 < gamma <= 1
		self.optim_fn = getattr(torch.optim, optim_fn)
		assert issubclass(self.optim_fn, torch.optim.Optimizer)

		self.location = location
		self.logger = Logger(f"{self.location}/train.log", name, verbose) #Already creates logger at init to test whether path works
		self.logger.log(f"Initialized {self.name}")

		self.evaluator = Evaluator(n_games=eval_games, max_time=max_time, scrambling_depths=scrambling_depths, logger=self.logger)
		self.evaluations = evaluations
		assert isinstance(self.evaluations, int) and 0 <= self.evaluations <= self.rollouts
		self.agent = agent
		assert isinstance(self.agent, DeepAgent)
		self.is2024 = is2024
		self.model_cfg = ModelConfig(architecture=arch)
		assert arch in ["fc", "res", "conv"]
		if arch == "conv": assert not get_is2024()
		assert isinstance(self.model_cfg, ModelConfig)

	def execute(self):

		# Clears directory to avoid clutter and mixing of experiments
		rmtree(self.location, ignore_errors=True)
		os.makedirs(self.location)

		# Sets representation
		store_repr()
		set_is2024(self.is2024)
		self.logger(f"Starting job:\n{self.name} with {'20x24' if get_is2024() else '6x8x6'} representation\nLocation {self.location}\nCommit: {get_commit()}")
		set_is2024(self.is2024)

		self.logger(f"Rough upper bound on total evaluation time during training: {self.evaluations*self.evaluator.approximate_time()/60:.2f} min")
		train = Train(self.rollouts,
				batch_size			= self.batch_size,
				rollout_games		= self.rollout_games,
				rollout_depth		= self.rollout_depth,
				loss_weighting		= self.loss_weighting,
				optim_fn			= self.optim_fn,
				lr					= self.lr,
				gamma				= self.gamma,
				agent				= self.agent,
				logger				= self.logger,
				evaluations			= self.evaluations,
				evaluator			= self.evaluator,
		)

		net = Model.create(self.model_cfg, self.logger).to(gpu)
		net, min_net = train.train(net)
		net.save(self.location)
		min_net.save(self.location, True)

		train.plot_training(self.location)
		train.plot_value_targets(self.location)
		train.plot_net_changes(self.location)
		np.save(f"{self.location}/rollouts.npy", train.train_rollouts)
		np.save(f"{self.location}/policy_losses.npy", train.policy_losses)
		np.save(f"{self.location}/value_losses.npy", train.value_losses)
		np.save(f"{self.location}/losses.npy", train.train_losses)
		np.save(f"{self.location}/evaluation_rollouts.npy", train.evaluations)
		np.save(f"{self.location}/evaluations.npy", train.eval_rewards)

		restore_repr()

		return train.train_rollouts, train.train_losses


if __name__ == "__main__":
	description = r"""

___________________________________________________________________
  /_/_/_/\	______ _      ______ _   _______ _____ _   __ _____
 /_/_/_/\/\	| ___ \ |     | ___ \ | | | ___ \_   _| | / //  ___|
/_/_/_/\/\/\| |_/ / |     | |_/ / | | | |_/ / | | | |/ / \ `--.
\_\_\_\/\/\/|    /| |     |    /| | | | ___ \ | | |    \  `--. \
 \_\_\_\/\/	| |\ \| |____ | |\ \| |_| | |_/ /_| |_| |\  \/\__/ /
  \_\_\_\/	\_| \_\_____/ \_| \_|\___/\____/ \___/\_| \_/\____/
__________________________________________________________________
Start one or more Reinforcement Learning training session(s)
on the Rubik's Cube using config or CLI arguments.
 """
	# SET SEED
	seedsetter()

	parser = Parser(options, description=description, name='train')
	jobs = [TrainJob(**settings) for settings in  parser.parse()]
	for job in jobs:
		job.execute()


