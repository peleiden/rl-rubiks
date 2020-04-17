import os
import json
from glob import glob as glob #glob
from ast import literal_eval

import numpy as np

from src.rubiks.solving.evaluation import Evaluator
from src.rubiks.solving.agents import Agent, DeepAgent
from src.rubiks.solving import search
from src.rubiks.model import Model, ModelConfig

from src.rubiks.utils.logger import Logger
from src.rubiks.utils.parse import Parser
from src.rubiks.utils import seedsetter

train_folders = sorted(glob('data/local_train2*')) #Stops working in the next millenium

options = {
	'location': {
		'default':  train_folders[0] if train_folders else '.',
		'help':	    "Location to search for model and save results.\nMust use location/<run_name>/model.pt structure.",
		'type':	    str,
	},
	'searcher': {
		'default':  'MCTS',
		'help':	    'Name of searcher for agent corresponding to searcher class in src.rubiks.solving.search',
		'type':	    str,
		'choices':  ['MCTS', 'PolicySearch','BFS', 'RandomDFS',],
	},
	'games': {
		'default':  10000,
		'help':	    'Number of games to play in evaluation for each agent.',
		'type':	    int,
	},
	'max_time': {
		'default':  60,
		'help':	    'Max searching time for agent',
		'type':	    int,
	},
	'scrambling': {
		'default':  '10 25',
		'help':	    'Two space-seperated integers (given in string delimeters such as --eval_scrambling "10 25")\n'
			    'Denoting interval of number of scramblings to be run.',
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
		'help':		'Whether or not graph search should be applied to MCTS to find the shortest path',
		'type':		literal_eval,
		'choices':	[True, False]
	},
	'policy_sample': {
		'default':	True,
		'help':		'Whether or not there should be sampled when using the PolicySearch agent',
		'type':		literal_eval,
		'choices':	[True, False]
	},
}

class EvalJob:
	def __init__(self,
			name: str,
			# Set by parser, should correspond to options above
			location: str,
			searcher: str,
			games: int,
			max_time: float,
			scrambling: str,
			mcts_c: float,
			mcts_nu: float,
			mcts_graph_search: bool,
			policy_sample: bool,

			# Currently not set by parser
			mcts_workers: int = 100,
			verbose: bool = True,
			in_subfolder: bool = False, #Should be true if there are multiple experiments
		):
		self.name = name
		self.location = location

		assert isinstance(games, int) and games
		assert max_time > 0
		assert scrambling[0] #dirty check for iter and not starting with 0 :)

		#Create evaluator
		self.logger = Logger(f"{self.location}/eval.log", name, verbose) #Already creates logger at init to test whether path works
		self.evaluator = Evaluator(n_games=games, max_time=max_time, scrambling_depths=scrambling, logger=self.logger)

		#Create agents
		searcher = getattr(search, searcher)
		assert issubclass(searcher, search.Searcher)

		if issubclass(searcher, search.DeepSearcher):
			self.agents, search_args = {}, {}

			#DeepSearchers need specific arguments
			if searcher == search.MCTS:
				assert all([mcts_c >= 0, mcts_nu >= 0, isinstance(mcts_graph_search, bool), isinstance(mcts_workers, int), mcts_workers > 0])
				search_args = {'c': mcts_c, 'nu': mcts_nu, 'search_graph': mcts_graph_search, 'workers': mcts_workers}
			elif searcher == search.PolicySearch:
				assert isinstance(policy_sample, bool)
				search_args = {'sample_policy': policy_sample}
			search_location = os.path.dirname(os.path.abspath(self.location)) if in_subfolder else self.location # Use parent folder, if parser has generated multiple folders
			#DeepSearchers might have to test multiple NN's
			for folder in glob(f"{search_location}/*/")+[search_location]:
				if not os.path.isfile(os.path.join(folder, 'model.pt')): continue
				searcher = searcher.from_saved(folder, **search_args)
				self.agents[f'{searcher} {"" if folder==search_location else folder}'] = DeepAgent(searcher)
			if not self.agents:
				raise FileNotFoundError(f"No model.pt found in folder or subfolder of {self.location}")
			self.logger.log(f"Loaded model from {search_location}")
		else:
			self.agents = {str(searcher): Agent(searcher)}

		self.logger.log(f"Initialized {self.name} with agents {' '.join(agent for agent in self.agents.keys())}")
		self.logger.log(f"TIME ESTIMATE: {len(self.agents)*self.evaluator.approximate_time()/60:.2f} min.\t(Rough upper bound)")

	def execute(self):
		self.logger.log(f"Beginning evaluator {self.name}")
		print(self.agents)
		for name, agent in self.agents.items():
			self.logger.section(f'Evaluationg agent {name}')
			res = self.evaluator.eval(agent)
			# self.evaluator.plot_eller_noget

			np.save(f"{self.location}/{name}_results.npy", res)



if __name__ == "__main__":
	description = r"""

___________________________________________________________________
  /_/_/_/\	______ _      ______ _   _______ _____ _   __ _____
 /_/_/_/\/\	| ___ \ |     | ___ \ | | | ___ \_   _| | / //  ___|
/_/_/_/\/\/\    | |_/ / |     | |_/ / | | | |_/ / | | | |/ / \ `--.
\_\_\_\/\/\/    |    /| |     |    /| | | | ___ \ | | |    \  `--. \
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
	for job in [EvalJob(**settings, in_subfolder=len(run_settings)>1) for settings in run_settings]:
		job.execute()

