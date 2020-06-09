import sys, os
from shutil import rmtree
from glob import glob as glob #glob

import json

import numpy as np
import torch

from librubiks.cube import get_is2024, with_used_repr, store_repr, restore_repr, set_is2024
from librubiks.utils import get_commit, Logger

from librubiks.model import Model, ModelConfig
from librubiks.train import Train

from librubiks.solving import agents
from librubiks.solving.agents import PolicySearch, ValueSearch, DeepAgent, Agent
from librubiks.solving.evaluation import Evaluator


class TrainJob:
	eval_games = 200  # Not given as arguments to __init__, as they should be accessible in runtime_estim
	max_time = 0.05
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
				 nn_init: str,
				 is2024: bool,
				 arch: str,
				 analysis: bool,
				 reward_method: str,

				 # Currently not set by argparser/configparser
				 agent = PolicySearch(net=None),
				 scrambling_depths: tuple = (10,),
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

		assert nn_init in ["glorot", "he"] or ( float(nn_init) or True ),\
				f"Initialization must be glorot, he or a number, but was {nn_init}"
		self.model_cfg = ModelConfig(architecture=arch, is2024=is2024, init=nn_init)

		self.analysis = analysis
		assert isinstance(self.analysis, bool)

		self.reward_method = reward_method
		assert self.reward_method in ["paper", "lapanfix", "schultzfix", "reward0"]

		assert arch in ["fc_small", "fc_big", "res_small", "res_big", "conv"]
		if arch == "conv": assert not self.is2024
		assert isinstance(self.model_cfg, ModelConfig)

	@with_used_repr
	def execute(self):

		# Clears directory to avoid clutter and mixing of experiments
		rmtree(self.location, ignore_errors=True)
		os.makedirs(self.location)

		# Sets representation
		self.logger.section(f"Starting job:\n{self.name} with {'20x24' if get_is2024() else '6x8x6'} representation\nLocation {self.location}\nCommit: {get_commit()}")

		train = Train(self.rollouts,
					  batch_size			= self.batch_size,
					  rollout_games			= self.rollout_games,
					  rollout_depth			= self.rollout_depth,
					  optim_fn				= self.optim_fn,
					  alpha_update			= self.alpha_update,
					  lr					= self.lr,
					  gamma					= self.gamma,
					  tau					= self.tau,
					  reward_method			= self.reward_method,
					  update_interval		= self.update_interval,
					  agent					= self.agent,
					  logger				= self.logger,
					  evaluation_interval	= self.evaluation_interval,
					  evaluator				= self.evaluator,
					  with_analysis			= self.analysis,
					  )
		self.logger(f"Rough upper bound on total evaluation time during training: {len(train.evaluation_rollouts)*self.evaluator.approximate_time()/60:.2f} min")

		net = Model.create(self.model_cfg, self.logger)
		net, min_net = train.train(net)
		net.save(self.location)
		if self.evaluation_interval:
			min_net.save(self.location, True)

		train.plot_training(self.location, name=self.name)
		analysispath = os.path.join(self.location, "analysis")
		datapath = os.path.join(self.location, "train-data")
		os.mkdir(datapath)
		os.mkdir(analysispath)

		if self.analysis:
			train.analysis.plot_substate_distributions(analysispath)
			train.analysis.plot_value_targets(analysispath)
			train.analysis.plot_net_changes(analysispath)
			train.analysis.visualize_first_states(analysispath)
			np.save(f"{datapath}/avg_target_values.npy", train.analysis.avg_value_targets)
			np.save(f"{datapath}/policy_entropies.npy", train.analysis.policy_entropies)
			np.save(f"{datapath}/substate_val_stds.npy", train.analysis.substate_val_stds)

		np.save(f"{datapath}/rollouts.npy", train.train_rollouts)
		np.save(f"{datapath}/policy_losses.npy", train.policy_losses)
		np.save(f"{datapath}/value_losses.npy", train.value_losses)
		np.save(f"{datapath}/losses.npy", train.train_losses)
		np.save(f"{datapath}/evaluation_rollouts.npy", train.evaluation_rollouts)
		np.save(f"{datapath}/evaluations.npy", train.sol_percents)

		return train.train_rollouts, train.train_losses

class EvalJob:
	is2024: bool

	def __init__(self,
				 name: str,
				 # Set by parser, should correspond to options in runeval
				 location: str,
				 use_best: bool,
				 agent: str,
				 games: int,
				 max_time: float,
				 max_states: int,
				 scrambling: str,
				 optimized_params: bool,
				 mcts_c: float,
				 mcts_graph_search: bool,
				 policy_sample: bool,
				 astar_lambda: float,
				 astar_expansions: int,
				 egvm_epsilon: float,
				 egvm_workers: int,
				 egvm_depth: int,

				 # Currently not set by parser
				 verbose: bool = True,
				 in_subfolder: bool = False,  # Should be true if there are multiple experiments
			 ):

		self.name = name
		self.location = location

		assert isinstance(games, int) and games
		assert max_time >= 0
		assert max_states >= 0
		assert max_time or max_states
		scrambling = range(*scrambling)
		assert scrambling[0] #dirty check for iter and not starting with 0 :)
		assert isinstance(optimized_params, bool)

		#Create evaluator
		self.logger = Logger(f"{self.location}/{self.name}.log", name, verbose) #Already creates logger at init to test whether path works
		self.evaluator = Evaluator(n_games=games, max_time=max_time, max_states=max_states, scrambling_depths=scrambling, logger=self.logger)

		#Create agents
		agent_string = agent
		agent = getattr(agents, agent_string)
		assert issubclass(agent, agents.Agent)

		if issubclass(agent, agents.DeepAgent):
			self.agents, self.reps, agents_args = {}, {}, {}

			#DeepAgents need specific arguments
			if agent == agents.MCTS:
				assert mcts_c >= 0, f"Exploration parameter c must be 0 or larger, not {mcts_c}"
				agents_args = { 'c': mcts_c, 'search_graph': mcts_graph_search }
			elif agent == agents.PolicySearch:
				assert isinstance(policy_sample, bool)
				agents_args = { 'sample_policy': policy_sample }
			elif agent == agents.AStar:
				assert isinstance(astar_lambda, float) and 0 <= astar_lambda <= 1, "AStar lambda must be float in [0, 1]"
				assert isinstance(astar_expansions, int) and astar_expansions >= 1 and (not max_states or astar_expansions < max_states) , "Expansions must be int < max states"
				agents_args = { 'lambda_': astar_lambda, 'expansions': astar_expansions }
			elif agent == agents.EGVM:
				assert isinstance(egvm_epsilon, float) and 0 <= egvm_epsilon <= 1, "EGVM epsilon must be float in [0, 1]"
				assert isinstance(egvm_workers, int) and egvm_workers >= 1, "Number of EGWM workers must a natural number"
				assert isinstance(egvm_depth, int) and egvm_depth >= 1, "EGWM depth must be a natural number"
				agents_args = { 'epsilon': egvm_epsilon, 'workers': egvm_workers, 'depth': egvm_depth }
			else:  # Non-parametric methods go brrrr
				agents_args = {}

			search_location = os.path.dirname(os.path.abspath(self.location)) if in_subfolder else self.location # Use parent folder, if parser has generated multiple folders
			# DeepAgent might have to test multiple NN's
			for folder in glob(f"{search_location}/*/") + [search_location]:
				if not os.path.isfile(os.path.join(folder, 'model.pt')): continue
				store_repr()
				with open(f"{folder}/config.json") as f:
					cfg = json.load(f)
				if optimized_params and agent in [agents.MCTS, agents.AStar]:
					parampath = os.path.join(folder, f'{agent_string}_params.json')
					if os.path.isfile(parampath):
						with open(parampath, 'r') as paramfile:
							agents_args = json.load(paramfile)
							if agent == agents.MCTS: agents_args['search_graph'] = mcts_graph_search
					else:
						self.logger.log(f"Optimized params was set to true, but no file {parampath} was found, proceding with arguments for this {agent_string}.")

				set_is2024(cfg["is2024"])
				agent = agent.from_saved(folder, use_best=use_best, **agents_args)
				key = f'{str(agent)}{"" if folder == search_location else " " + os.path.basename(folder.rstrip(os.sep))}'

				self.reps[key] = cfg["is2024"]
				self.agents[key] = agent
				restore_repr()

			if not self.agents:
				raise FileNotFoundError(f"No model.pt found in folder or subfolder of {self.location}")
			self.logger.log(f"Loaded model from {search_location}")

		else:
			agent = agent()
			self.agents = {agent: agent}
			self.reps = {agent: True}

		self.agent_results = {}
		self.logger.log(f"Initialized {self.name} with agents {', '.join(str(s) for s in self.agents)}")
		self.logger.log(f"TIME ESTIMATE: {len(self.agents) * self.evaluator.approximate_time() / 60:.2f} min.\t(Rough upper bound)")

	def execute(self):
		self.logger.log(f"Beginning evaluator {self.name}\nLocation {self.location}\nCommit: {get_commit()}")
		for (name, agent), representation in zip(self.agents.items(), self.reps.values()):
			self.is2024 = representation
			self.agent_results[name] = self._single_exec(name, agent)

	@with_used_repr
	def _single_exec(self, name: str, agent: Agent):
		self.logger.section(f'Evaluationg agent {name}')
		res, states, times = self.evaluator.eval(agent)
		subfolder = os.path.join(self.location, "evaluation_results")
		os.makedirs(subfolder, exist_ok=True)
		paths = [os.path.join(subfolder, f"{name}_results.npy"), os.path.join(subfolder, f"{name}_states_seen.npy"), os.path.join(subfolder, f"{name}_playtimes.npy")]
		np.save(paths[0], res)
		np.save(paths[1], states)
		np.save(paths[2], times)
		self.logger.log("Saved evaluation results to\n" + "\n".join(paths))
		return res, states, times

	@staticmethod
	def plot_all_jobs(jobs: list, save_location: str):
		results, states, times, settings = dict(), dict(), dict(), dict()
		for job in jobs:
			for agent, (result, states_, times_) in job.agent_results.items():
				key = agent if len(jobs) == 1 else f"{job.name} - {agent}"
				results[key] = result
				states[key] = states_
				times[key] = times_
				settings[key] = {
					'n_games': job.evaluator.n_games,
					'max_time': job.evaluator.max_time,
					'scrambling_depths': job.evaluator.scrambling_depths
				}
		savepaths = Evaluator.plot_evaluators(results, states, times, settings, save_location)
		job.logger(f"Saved plots to {savepaths}")

