import sys

from argparse import ArgumentParser
from configparser import ConfigParser

import torch

import src.rubiks.solving.agents as agents
from src.rubiks.utils.logger import Logger
from src.rubiks.utils import cpu, gpu
from src.rubiks import get_repr, set_repr
from src.rubiks.model import Model, ModelConfig
from src.rubiks.train import Train
from src.rubiks.solving.evaluation import Evaluator


defaults  = {
	'rollouts': 1000,
	'location': 'local_train',
	'rollout_games': 1000,
	'rollout_depth': 100,
	'batch_size': 50,
	'lr': 1e-5,
	'optim_fn': 'RMSprop',
	'agent': 'PolicyCube',
	}

class TrainJob:
	def __init__(self,
			jobname: str,
			# Set by parser, should correspond to values in `default` above and defaults can be controlled there
			location: str,
			rollouts: int,
			rollout_games: int,
			rollout_depth: int,
			batch_size: int,
			lr: float,
			optim_fn: str,
			agent: str,

			# Currently not set by argparser/configparser
			is2024: bool = True,
			verbose: bool = True,
			model_cfg: ModelConfig = ModelConfig(batchnorm=False),
			):
		self.jobname = jobname
		assert isinstance(self.jobname, str)

		self.rollouts = rollouts
		assert self.rollouts > 0
		self.rollout_games = rollout_games
		assert self.rollout_games > 0
		self.rollout_depth = rollout_depth
		assert rollout_depth > 0

		self.batch_size = batch_size
		assert self.batch_size > 0 and self.batch_size <= self.rollout_games * self.rollout_depth
		self.lr = lr
		assert float(lr) and lr <= 1

		self.optim_fn = getattr(torch.optim, optim_fn)
		assert issubclass(self.optim_fn, torch.optim.Optimizer)

		self.agent = getattr(agents, agent)
		assert issubclass(self.agent, agents.DeepAgent)

		self.location = location
		self.logger = Logger(f"{self.location}/{self.jobname}.log", jobname, verbose) #Already creates logger at init to test whether path works
		self.logger.log(f"Initialized {self.jobname}")


		self.is2024 = is2024
		self.model_cfg = model_cfg
		assert isinstance(self.model_cfg, ModelConfig)

	def execute(self):
		self.logger(f"Starting job:\n{self.jobname}")
		ini_repr = get_repr()
		set_repr(self.is2024)

		# Training
		self.logger.section()
		train = Train(self.rollouts,
				batch_size	=self.batch_size,
				rollout_games	=self.rollout_games,
				rollout_depth	=self.rollout_depth,
				optim_fn	=self.optim_fn,
				lr		=self.lr,
				agent		=self.agent,
				logger		=self.logger,
		)



		net = Model(self.model_cfg, self.logger).to(gpu)
		net = train.train(net)
		net.save(self.location)

		train.plot_training(self.location)

		# Evaluation
#		logger.section()
#		evaluator = Evaluator(**job.eval_args, logger=logger)
#		for agent_fn in job.agents:
#			agent = agent_fn(net)

		# TODO: Finish implementing evaluation and return both training and evaluation results

		set_repr(ini_repr)

		return train.train_rollouts, train.train_losses

def parse(defaults: dict):
	# First off: A seperate parser for the configuration file
	config_parser = ArgumentParser(add_help = False)
	config_parser.add_argument('--config', help="Location of config file", metavar="FILE")
	args, remaining_args = config_parser.parse_known_args()

	#Parser for the rest of the options is added
	parser = ArgumentParser(
			description = 'Start one or more training sessions of the DeepCube agent using config or CLI arguments. If multiple runs are specified in config AND CLI arguments are given, CLI arguments overwrite settings for all runs',
	parents = [config_parser])

	parser.add_argument('--rollouts', help="Number of rollouts: Number of passes of ADI+parameter update", type=int)
	parser.add_argument('--location', help="Save location for logs and plots", type=str)
	parser.add_argument('--rollout_games', help="Number of games in ADI in each rollout", type=int)
	parser.add_argument('--rollout_depth', help="Number of scramblings applied to each game in ADI", type=int)
	parser.add_argument('--batch_size', help="Number of training examples to be used at the same time in parameter update", type=int)
	parser.add_argument('--lr', help="Learning rate of parameter update", type=float)
	parser.add_argument('--optim_fn', help="String corresponding to a class in torch.optim", type=str)
	parser.add_argument('--agent', help="String corresponding to a deepagent class in src.rubiks.solving.agents", type=str, choices = ["PolicyCube", "DeepCube"])

	jobs = list()
	if args.config:
		config = ConfigParser()
		config.read([args.config])
		# If user sets a DEFAULT section, then this overwrites the defaults
		defaults = {**defaults, **dict(config.items("DEFAULT"))}

		# Each section corresponds to a job
		for jobname in config.sections():
			options = {**defaults, **dict(config.items(jobname))}
			parser.set_defaults(**options)

			job_args = parser.parse_args(remaining_args)
			save_location = job_args.location #Save for dump at the end
			job_args.location = f"{job_args.location}/{jobname}" # Give unique location to each run
			del job_args.config
			jobs.append(TrainJob(jobname, **vars(job_args)))
		with open(f"{save_location}/used_config.ini", 'w') as f: config.write(f)
	# If no config was added or config of only defaults were  added, run from CLI/defaults.
	if (not args.config) or (not config.sections()):
		parser.set_defaults(**defaults)
		args = parser.parse_args(remaining_args)
		save_location = args.location
		del args.config
		jobs.append(TrainJob("Training", **vars(args)))

	# For reproduceability: Save config file and arguments
	with open(f"{save_location}/used_config.ini", 'a') as f: f.write(f"#{' '.join(sys.argv)}")
	return jobs



if __name__ == "__main__":

	jobs = parse(defaults)
	for job in jobs:
		job.execute()
