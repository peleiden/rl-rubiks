import sys, os

from ast import literal_eval

import numpy as np
import torch

from src.rubiks.solving import search
from src.rubiks.utils import seedsetter
from src.rubiks.utils.parse import Parser
from src.rubiks.utils.ticktock import get_timestamp
from src.rubiks.utils.logger import Logger
from src.rubiks import cpu, gpu, get_repr, set_repr, store_repr, restore_repr
from src.rubiks.model import Model, ModelConfig
from src.rubiks.train import Train
from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.solving.agents import DeepAgent

options = {
	'location': {
		'default':  'data/local_train'+get_timestamp(for_file=True),
		'help':	    "Save location for logs and plots",
		'type':	    str,
	},
	'rollouts': {
		'default':  1000,
		'help':	    'Number of passes of ADI+parameter update',
		'type':	    int,
	},
	'rollout_games': {
		'default':  1000,
		'help':	    'Number of games in ADI in each rollout',
		'type':	    int,
	},
	'rollout_depth': {
		'default':  100,
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
	'optim_fn': {
		'default':  'RMSprop',
		'help':	    'Name of optimization function corresponding to class in torch.optim',
		'type':	    str,
	},
	'searcher': {
		'default':  'MCTS',
		'help':	    'Name of searcher for agent corresponding to deepseacher class in src.rubiks.solving.search',
		'type':	    str,
		'choices':  ['MCTS', 'PolicySearch'],
	},
	'evaluations': {
		'default':  20,
		'help':	    'Number of evaluations during training. Some settings for these hard coded',
		'type':	    int,
	},
	'train_eval_games': {
		'default':  150,
		'help':	    'Number of games used for each evaluation during training',
		'type':	    int,
	},
	'final_evals': {
		'default':  10000,
		'help':	    'Number of games in final evaluation',
		'type':	    int,
	},
	'eval_max_time': {
		'default':  60,
		'help':	    'Max searching time for agent in the FINAL evaluation',
		'type':	    int,
	},
	'eval_scrambling': {
		'default':  '10 25',
		'help':	    'Two space-seperated integers (given in string delimeters such as --eval_scrambling "10 20")\ndenoting interval'
			    ' of number of scramblings to be run in final evaluation.\nIn eval during training, the mean of these is used',
		#Ugly way to define list of two numbers
		'type':	    lambda args: [int(args.split()[0]), int(args.split()[1])],
	},
	'mcts_c': {
		'default':	1,
		'help':		'Exploration parameter c for MCTS',
		'type':		float,
	},
	'mcts_nu': {
		'default':	1,
		'help':		'Virtual loss nu for MCTS',
		'type':		float,
	},
	'mcts_graph_search': {
		'default':	True,
		'help':		'Whether or not graph search should be applied to MCTS in the post training evaluation to find the shortest path',
		'type':		literal_eval,
		'choices':	[True, False]
	},
	'is2024': {
		'default':  True,
		'help':	    'True for 20x24 Rubiks representation and False for 6x8x6',
		'type':	    literal_eval,
		'choices':  [True, False],
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
			optim_fn: str,
			searcher: str,
			evaluations: int,
			train_eval_games: int,
			eval_max_time: int,
			eval_scrambling: list,
			final_evals: int,
			mcts_c: float,
			mcts_nu: float,
			mcts_graph_search: bool,
			is2024: bool,

			# Currently not set by argparser/configparser
			verbose: bool = True,
			model_cfg: ModelConfig = ModelConfig(batchnorm=False),
			max_train_eval_time = 1,
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

		self.optim_fn = getattr(torch.optim, optim_fn)
		assert issubclass(self.optim_fn, torch.optim.Optimizer)

		searcher = getattr(search, searcher)
		assert issubclass(searcher, search.DeepSearcher)
		if searcher == search.MCTS:
			assert all([mcts_c >= 0, mcts_nu >= 0, isinstance(mcts_graph_search, bool)])
			self.train_agent = DeepAgent(search.MCTS(None, mcts_c, mcts_nu, False))
			self.eval_agent = DeepAgent(search.MCTS(None, mcts_c, mcts_nu, mcts_graph_search))
		elif searcher == search.PolicySearch:
			self.train_agent = DeepAgent(search.PolicySearch(None, True))
			self.eval_agent = DeepAgent(search.PolicySearch(None, True))

		self.evaluations = evaluations
		assert self.evaluations <= self.rollouts
		self.eval_max_time = eval_max_time
		assert float(eval_max_time)
		self.eval_scrambling = range(*eval_scrambling)
		assert int(np.mean(self.eval_scrambling))
		self.final_evals = final_evals
		assert isinstance(self.final_evals, int)

		self.train_eval_games = train_eval_games
		assert isinstance(self.train_eval_games, int) and self.train_eval_games > 0
		self.max_train_eval_time = max_train_eval_time
		assert self.max_train_eval_time > 0

		self.location = location
		self.logger = Logger(f"{self.location}/{self.name}.log", name, verbose) #Already creates logger at init to test whether path works
		self.logger.log(f"Initialized {self.name}")

		self.is2024 = is2024
		self.model_cfg = model_cfg
		assert isinstance(self.model_cfg, ModelConfig)

	def execute(self):
		self.logger(f"Starting job:\n{self.name}")
		store_repr()
		set_repr(self.is2024)

		# Training
		self.logger.section()

		if self.final_evals:
			final_evaluator = Evaluator(n_games=self.final_evals, max_time=self.eval_max_time, scrambling_depths=self.eval_scrambling, logger=self.logger)
			self.logger(f"Rough upper bound on final evaluation time: {final_evaluator.approximate_time()/60:.2f} min.")

		train_scramble = int(np.mean(self.eval_scrambling))
		train_evaluator = Evaluator(n_games=self.train_eval_games, max_time=min(self.eval_max_time, self.max_train_eval_time), scrambling_depths=[train_scramble], logger=self.logger)
		self.logger(f"Rough upper bound on total evaluation time during training: {self.evaluations*train_evaluator.approximate_time()/60:.2f} min")
		train = Train(self.rollouts,
				batch_size		= self.batch_size,
				rollout_games	= self.rollout_games,
				rollout_depth	= self.rollout_depth,
				loss_weighting	= self.loss_weighting,
				optim_fn		= self.optim_fn,
				lr				= self.lr,
				agent			= self.train_agent,
				logger			= self.logger,
				evaluations		= self.evaluations,
				evaluator		= train_evaluator,
		)

		net = Model(self.model_cfg, self.logger).to(gpu)
		net, min_net = train.train(net)
		net.save(self.location)
		min_net.save(self.location, True)

		train.plot_training(self.location)
		train.plot_value_targets(self.location)
		np.save(f"{self.location}/rollouts.npy", train.train_rollouts)
		np.save(f"{self.location}/policy_losses.npy", train.policy_losses)
		np.save(f"{self.location}/value_losses.npy", train.value_losses)
		np.save(f"{self.location}/losses.npy", train.train_losses)
		np.save(f"{self.location}/evaluation_rollouts.npy", train.evaluations)
		np.save(f"{self.location}/evaluations.npy", train.eval_rewards)

		if self.final_evals:
			# Evaluation
			self.logger.section()
			self.eval_agent.update_net(net)
			final_evaluator.eval(self.eval_agent)

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
	parser = Parser(options, description=description)
	jobs = [TrainJob(**settings) for settings in  parser.parse()]
	for job in jobs:
		job.execute()
