import sys

from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import torch

import src.rubiks.solving.agents as agents
from src.rubiks.utils.logger import Logger
from src.rubiks import cpu, gpu, get_repr, set_repr, store_repr, restore_repr
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
	'evaluations': 20,
	'eval_max_time': 60,
	'eval_scrambling': '10 25',
	'final_evals': 10000,
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
			evaluations: int,
			eval_max_time: int,
			eval_scrambling: list,
			final_evals: int,

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

		self.evaluations = evaluations
		assert self.evaluations <= self.rollouts
		self.eval_max_time = eval_max_time
		assert float(eval_max_time)
		self.eval_scrambling = range(*eval_scrambling)
		assert int(np.mean(self.eval_scrambling))
		self.final_evals = final_evals
		assert isinstance(self.final_evals, int)

		self.location = location
		self.logger = Logger(f"{self.location}/{self.jobname}.log", jobname, verbose) #Already creates logger at init to test whether path works
		self.logger.log(f"Initialized {self.jobname}")


		self.is2024 = is2024
		self.model_cfg = model_cfg
		assert isinstance(self.model_cfg, ModelConfig)

	def execute(self):
		self.logger(f"Starting job:\n{self.jobname}")
		store_repr()
		set_repr(self.is2024)

		# Training
		self.logger.section()

		train_scramble = int(np.mean(self.eval_scrambling))

		train_evaluator = Evaluator(n_games=int(np.ceil(1/4*self.rollout_games)), max_time=self.eval_max_time, scrambling_depths=[train_scramble], logger=self.logger)
		train = Train(self.rollouts,
				batch_size	=self.batch_size,
				rollout_games	=self.rollout_games,
				rollout_depth	=self.rollout_depth,
				optim_fn	=self.optim_fn,
				lr		=self.lr,
				agent		=self.agent,
				logger		=self.logger,
				evaluations	=self.evaluations,
				evaluator	=train_evaluator,
		)



		net = Model(self.model_cfg, self.logger).to(gpu)
		net = train.train(net)
		net.save(self.location)

		train.plot_training(self.location)

		# Evaluation
		self.logger.section()
		evaluator = Evaluator(n_games=self.final_evals, max_time=self.eval_max_time, scrambling_depths=self.eval_scrambling, logger=self.logger)
		evaluator.eval(self.agent(net))

		restore_repr()

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
	parser.add_argument('--evaluations', help="Number of evaluations (each consisting of 1/4 og rollout_games) to be done during training", type=int)
	parser.add_argument('--eval_max_time', help="Max time (seconds) for each game for the agent", type=int)
	intlist_validator = lambda args: [int(args.split()[0]), int(args.split()[1])] #Ugly way to define list of two numbers
	parser.add_argument('--eval_scrambling', help="Two space-seperated integers (given in string delimeters, such as --eval scrambling '10 20') denoting interval of number of scramblings to be run in evaluation. In evaluation during training, the mean of these is used", type=intlist_validator)
	parser.add_argument('--final_evals', help="Number of games to be done in the evaluation after the training", type=int)

	jobs = list()
	with_config = False
	if args.config:
		with_config = True
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

	# If no config was added or config of only defaults were  added, run from CLI/defaults.
	if (not args.config) or (not config.sections()):
		parser.set_defaults(**defaults)
		args = parser.parse_args(remaining_args)
		save_location = args.location
		del args.config
		jobs.append(TrainJob("Training", **vars(args)))

	# For reproduceability: Save config file and arguments
	if with_config:
		with open(f"{save_location}/used_config.ini", 'w') as f: config.write(f)

	with open(f"{save_location}/used_config.ini", 'a') as f: f.write(f"#{' '.join(sys.argv)}")
	return jobs



if __name__ == "__main__":

	jobs = parse(defaults)
	for job in jobs:
		job.execute()
