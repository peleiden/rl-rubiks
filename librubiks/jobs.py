import sys, os
from shutil import rmtree
from glob import glob as glob #glob

import json

import numpy as np
import torch

from librubiks import gpu, get_is2024, with_used_repr, store_repr, restore_repr, set_is2024
from librubiks.utils import get_commit, Logger

from librubiks.model import Model, ModelConfig
from librubiks.train import Train

from librubiks.solving.evaluation import Evaluator
from librubiks.solving.agents import Agent, DeepAgent
from librubiks.solving.search import PolicySearch, ValueSearch

import librubiks.solving.search as search

class TrainJob:
	eval_games = 200  # Not given as arguments to __init__, as they should be accessible in runtime_estim
	max_time = 0.01
	is2024: bool

	def __init__(self,
				 name: str,
				 # Set by parser, should correspond to options in runtrain
				 location: str,
				 rollouts: int,
				 rollout_games: int,
				 rollout_depth: int,
				 batch_size: int,
				 alpha_update: float,
				 lr: float,
				 gamma: float,
				 tau: float,
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
		self.tau = tau
		assert 0 < tau <= 1
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
					  tau				= self.tau,
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
			train.analysis.visualize_first_states(self.location)
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

class EvalJob:
	is2024: bool

	def __init__(self,
			name: str,
			# Set by parser, should correspond to options in runeval
			location: str,
			searcher: str,
			games: int,
			max_time: float,
			max_states: int,
			scrambling: str,
			mcts_c: float,
			mcts_nu: float,
			mcts_graph_search: bool,
			mcts_workers: int,
			mcts_policy_type: str,
			policy_sample: bool,

			# Currently not set by parser
			verbose: bool = True,
			in_subfolder: bool = False, # Should be true if there are multiple experiments
		):
		self.name = name
		self.location = location


		assert isinstance(games, int) and games
		assert max_time > 0
		assert max_states >= 0
		assert max_time or max_states
		scrambling = range(*scrambling)
		assert scrambling[0] #dirty check for iter and not starting with 0 :)

		#Create evaluator
		self.logger = Logger(f"{self.location}/{self.name}.log", name, verbose) #Already creates logger at init to test whether path works
		self.evaluator = Evaluator(n_games=games, max_time=max_time, max_states=max_states, scrambling_depths=scrambling, logger=self.logger)

		#Create agents
		searcher = getattr(search, searcher)
		assert issubclass(searcher, search.Searcher)

		if issubclass(searcher, search.DeepSearcher):
			self.agents, self.reps, search_args = {}, {}, {}

			#DeepSearchers need specific arguments
			if searcher == search.MCTS:
				assert mcts_c >= 0 and mcts_nu >= 0\
					and isinstance(mcts_workers, int) and mcts_workers > 0\
					and mcts_policy_type in ["p", "v", "w"]
				search_args = {'c': mcts_c, 'nu': mcts_nu,  'search_graph': mcts_graph_search, 'workers': mcts_workers, 'policy_type': mcts_policy_type}
			elif searcher == search.PolicySearch:
				assert isinstance(policy_sample, bool)
				search_args = {'sample_policy': policy_sample}
			else:  # Non-parametric methods go brrrr
				search_args = {}

			search_location = os.path.dirname(os.path.abspath(self.location)) if in_subfolder else self.location # Use parent folder, if parser has generated multiple folders
			# DeepSearchers might have to test multiple NN's
			for folder in glob(f"{search_location}/*/")+[search_location]:
				if not os.path.isfile(os.path.join(folder, 'model.pt')): continue
				store_repr()
				with open(f"{folder}/config.json") as f:
					cfg = json.load(f)

				set_is2024(cfg["is2024"])
				searcher = searcher.from_saved(folder, **search_args)
				key = f'{str(searcher)} {"" if folder==search_location else os.path.basename(folder.rstrip(os.sep))}'

				self.reps[key] = cfg["is2024"]
				self.agents[key] = DeepAgent(searcher)
				restore_repr()

			if not self.agents:
				raise FileNotFoundError(f"No model.pt found in folder or subfolder of {self.location}")
			self.logger.log(f"Loaded model from {search_location}")

		else:
			searcher = searcher()
			self.agents = {searcher: Agent(searcher)}
			self.reps = {searcher: True}

		self.agent_results = {}
		self.logger.log(f"Initialized {self.name} with agents {' '.join(str(agent) for agent in self.agents)}")
		self.logger.log(f"TIME ESTIMATE: {len(self.agents)*self.evaluator.approximate_time()/60:.2f} min.\t(Rough upper bound)")

	def execute(self):
		self.logger.log(f"Beginning evaluator {self.name}\nLocation {self.location}\nCommit: {get_commit()}")
		for (name, agent), representation in zip(self.agents.items(), self.reps.values()):
			self.is2024 = representation
			self.agent_results[name] = self._single_exec(name, agent)


	@with_used_repr
	def _single_exec(self, name, agent):
		self.logger.section(f'Evaluationg agent {name}')
		res = self.evaluator.eval(agent)
		np.save(f"{self.location}/{name}_results.npy", res)
		return res

	@staticmethod
	def plot_all_jobs(jobs: list, save_location: str):
		results, settings = dict(), list()
		for job in jobs:
			for agent, result in job.agent_results.items():
				key = agent if len(jobs) == 1 else f"{job.name} {agent}"
				results[key] = result
				settings.append(
					{
						'n_games': job.evaluator.n_games,
						'max_time': job.evaluator.max_time,
						'scrambling_depths': job.evaluator.scrambling_depths
					}
				)
		savepaths = Evaluator.plot_evaluators(results, save_location, settings)
		for job in jobs: job.logger(f"Saved plots to {savepaths}")

