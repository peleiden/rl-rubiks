import sys, os
from shutil import rmtree

from ast import literal_eval

import numpy as np
import torch

from src.rubiks.utils import seedsetter, get_commit, get_timestamp
from src.rubiks.utils.parse import Parser
from src.rubiks.utils.logger import Logger

from src.rubiks import gpu, get_is2024, with_used_repr
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
	'batch_size': {
		'default':  50,
		'help':	    'Number of training examples to be used in each parameter update',
		'type':	    int
	},
	'alpha_update': {
		'default':  0,
		'help':	    'alpha is set to alpha + alpha_update 100 times during training, though never more than 1. 0 for weighted and 1 for unweighted',
		'type':	    float,
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
	'update_interval': {
		'default':	50,
		'help':		'How often alpha and lr are updated. First update is performed when rollout == update_interval. Set to 0 for never',
		'type':		int,
	},
	'optim_fn': {
		'default':  'RMSprop',
		'help':	    'Name of optimization function corresponding to class in torch.optim',
		'type':	    str,
	},
	'evaluation_interval': {
		'default':  100,
		'help':	    'An evaluation is performed every evaluation_interval rollouts. Set to 0 for never',
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
	'analysis': {
		'default': False,
		'help':	   'If true, analysis of model changes, value and loss behaviour is done in each rollout and ADI pass',
		'type':	    literal_eval,
		'choices':  [True, False],
	},
}



class TrainJob:
	eval_games = 200  # Not given as arguments to __init__, as they should be accessible in runtime_estim
	max_time = 0.01
	is2024: bool

	def __init__(self,
				 name: str,
				 # Set by parser, should correspond to values in `options`  above and defaults can be controlled there
				 location: str,
				 rollouts: int,
				 rollout_games: int,
				 rollout_depth: int,
				 batch_size: int,
				 alpha_update: float,
				 lr: float,
				 gamma: float,
				 update_interval: int,
				 optim_fn: str,
				 evaluation_interval: int,
				 is2024: bool,
				 arch: str,
				 analysis: bool,


				 # Currently not set by argparser/configparser
				 agent = DeepAgent(PolicySearch(None, True)),
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
		self.batch_size = batch_size
		assert 0 < self.batch_size <= self.rollout_games * self.rollout_depth

		self.alpha_update = alpha_update
		assert 0 <= alpha_update <= 1
		self.lr = lr
		assert float(lr) and lr <= 1
		self.gamma = gamma
		assert 0 < gamma <= 1
		self.update_interval = update_interval
		assert isinstance(self.update_interval, int) and 0 <= self.update_interval
		self.optim_fn = getattr(torch.optim, optim_fn)
		assert issubclass(self.optim_fn, torch.optim.Optimizer)

		self.location = location
		self.logger = Logger(f"{self.location}/train.log", name, verbose) #Already creates logger at init to test whether path works
		self.logger.log(f"Initialized {self.name}")

		self.evaluator = Evaluator(n_games=self.eval_games, max_time=self.max_time, scrambling_depths=scrambling_depths, logger=self.logger)
		self.evaluation_interval = evaluation_interval
		assert isinstance(self.evaluation_interval, int) and 0 <= self.evaluation_interval
		self.agent = agent
		assert isinstance(self.agent, DeepAgent)
		self.is2024 = is2024
		self.model_cfg = ModelConfig(architecture=arch, is2024=is2024)

		self.analysis = analysis
		assert isinstance(self.analysis, bool)

		###################
		# Temporary change of residual architecture to check for difference
		if arch == 'res':
			self.model_cfg.part_sizes = [512]
			self.model_cfg.res_size = 1000
			self.model_cfg.res_blocks = 2
			self.model_cfg.shared_sizes = [1000]
		##################
		assert arch in ["fc", "res", "conv"]
		if arch == "conv": assert not self.is2024
		assert isinstance(self.model_cfg, ModelConfig)

	@with_used_repr
	def execute(self):

		# Clears directory to avoid clutter and mixing of experiments
		rmtree(self.location, ignore_errors=True)
		os.makedirs(self.location)

		# Sets representation
		self.logger(f"Starting job:\n{self.name} with {'20x24' if get_is2024() else '6x8x6'} representation\nLocation {self.location}\nCommit: {get_commit()}")

		train = Train(self.rollouts,
					  batch_size			= self.batch_size,
					  rollout_games			= self.rollout_games,
					  rollout_depth			= self.rollout_depth,
					  optim_fn				= self.optim_fn,
					  alpha_update			= self.alpha_update,
					  lr					= self.lr,
					  gamma					= self.gamma,
					  update_interval		= self.update_interval,
					  agent					= self.agent,
					  logger				= self.logger,
					  evaluation_interval	= self.evaluation_interval,
					  evaluator				= self.evaluator,
					  with_analysis			= self.analysis,
				  )
		self.logger(f"Rough upper bound on total evaluation time during training: {len(train.evaluation_rollouts)*self.evaluator.approximate_time()/60:.2f} min")

		net = Model.create(self.model_cfg, self.logger).to(gpu)
		net, min_net = train.train(net)
		net.save(self.location)
		min_net.save(self.location, True)

		train.plot_training(self.location)
		datapath = os.path.join(self.location, "train-data")
		os.mkdir(datapath)

		if self.analysis:
			train.analysis.plot_substate_distributions(self.location)
			train.analysis.plot_value_targets(self.location)
			train.analysis.plot_net_changes(self.location)
			np.save(f"{datapath}/avg_target_values.npy", train.analysis.avg_value_targets)
			np.save(f"{datapath}/policy_entropies.npy", train.analysis.policy_entropies)
			np.save(f"{datapath}/substate_val_stds.npy", train.analysis.substate_val_stds)

		np.save(f"{datapath}/rollouts.npy", train.train_rollouts)
		np.save(f"{datapath}/policy_losses.npy", train.policy_losses)
		np.save(f"{datapath}/value_losses.npy", train.value_losses)
		np.save(f"{datapath}/losses.npy", train.train_losses)
		np.save(f"{datapath}/evaluation_rollouts.npy", train.evaluation_rollouts)
		np.save(f"{datapath}/evaluations.npy", train.eval_rewards)

		return train.train_rollouts, train.train_losses


def clean_dir(loc: str):
	"""
	Cleans a training directory created by runtrain
	All except the config file is removed
	"""
	with open(f"{loc}/train_config.ini") as f:
		content = f.read()
	rmtree(loc)
	os.mkdir(loc)
	with open(f"{loc}/train_config.ini", "w") as f:
		f.write(content)


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
	clean_dir(parser.save_location)
	for job in jobs:
		job.execute()


