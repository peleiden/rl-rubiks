import os
import json
from glob import glob as glob #glob
from ast import literal_eval

import numpy as np

from src.rubiks import store_repr, set_is2024, restore_repr, with_used_repr
from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.solving.agents import Agent, DeepAgent
from src.rubiks.solving import search
from src.rubiks.model import Model, ModelConfig

from src.rubiks.utils import seedsetter, get_commit
from src.rubiks.utils.logger import Logger
from src.rubiks.utils.parse import Parser

train_folders = sorted(glob('data/local_train2*')) #Stops working in the next millenium

options = {
	'location': {
		'default':  train_folders[-1] if train_folders else '.',
		'help':	    "Location to search for model and save results.\nMust use location/<run_name>/model.pt structure.",
		'type':	    str,
	},
	'searcher': {
		'default':  'MCTS',
		'help':	    'Name of searcher for agent corresponding to searcher class in src.rubiks.solving.search',
		'type':	    str,
		'choices':  ['MCTS', 'PolicySearch','BFS', 'RandomDFS', 'AStar',],
	},
	'games': {
		'default':  10,
		'help':	    'Number of games to play in evaluation for each depth, for each agent.',
		'type':	    int,
	},
	'max_time': {
		'default':  30,
		'help':	    'Max searching time for agent per configuration. 0 for unlimited',
		'type':	    literal_eval,
	},
	'max_states': {
		'default':  0,
		'help':	    'Max number of searched states for agent per configuration. 0 for unlimited',
		'type':	    lambda arg: int(float(arg)),
	},
	'scrambling': {
		'default':  '10 25',
		'help':	    'Two space-seperated integers (given in string delimeters such as --eval_scrambling "10 25")\n'
			    'Denoting interval of number of scramblings to be run.',
		#Ugly way to define list of two numbers
		'type':	    lambda args: [int(args.split()[0]), int(args.split()[1])],
	},
	'mcts_c': {
		'default':	0.6,
		'help':		'Exploration parameter c for MCTS',
		'type':		float,
	},
	'mcts_nu': {
		'default':	.005,
		'help':		'Virtual loss nu for MCTS',
		'type':		float,
	},
	'mcts_graph_search': {
		'default':	True,
		'help':		'Whether or not graph search should be applied to MCTS to find the shortest path',
		'type':		literal_eval,
		'choices':	[True, False],
	},
	'mcts_workers': {
		'default':	10,
		'help':		'Number of sequential workers in MCTS',
		'type':		int,
	},
	'policy_sample': {
		'default':	False,
		'help':		'Whether or not there should be sampled when using the PolicySearch agent',
		'type':		literal_eval,
		'choices':	[True, False],
	},
}

class EvalJob:
	is2024: bool

	def __init__(self,
			name: str,
			# Set by parser, should correspond to options above
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
			policy_sample: bool,

			# Currently not set by parser
			verbose: bool = True,
			in_subfolder: bool = False, # Should be true if there are multiple experiments
		):
		self.name = name
		self.location = location

		assert isinstance(games, int) and games
		assert max_time >= 0
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
					and isinstance(mcts_graph_search, bool)\
					and isinstance(mcts_workers, int) and mcts_workers > 0
				search_args = {'c': mcts_c, 'nu': mcts_nu, 'search_graph': mcts_graph_search, 'workers': mcts_workers}
			elif searcher == search.PolicySearch:
				assert isinstance(policy_sample, bool)
				search_args = {'sample_policy': policy_sample}
			elif searcher == search.AStar:
				search_args = {}  # Non-parametric method goes brrrr
			else: raise Exception(f"Kwargs have not been prepared for the DeepSearcher {searcher}")

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
		for i, job in enumerate(jobs): job.logger(f"Saved plots to {savepaths}")

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
Evaluate Rubiks agents using config or CLI arguments. If no location
is given, data/local_train with newest name is used. If the location
contains multiple neural networks, the deepsearchers are evalued for
each of them.
 """
	# SET SEED
	seedsetter()

	parser = Parser(options, description=description, name='eval')
	run_settings = parser.parse()
	jobs = [EvalJob(**settings, in_subfolder=len(run_settings)>1) for settings in run_settings]

	for job in jobs: job.execute()
	EvalJob.plot_all_jobs(jobs, parser.save_location)

