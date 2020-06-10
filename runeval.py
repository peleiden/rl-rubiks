from glob import glob as glob #glob

from ast import literal_eval

from librubiks.utils import Parser, set_seeds
from librubiks.jobs import EvalJob

train_folders = sorted(glob('data/local_train2*')) #Stops working in the next millenium

###
# Should correspond to arguments in EvalJob
###
options = {
	'location': {
		'default':  train_folders[-1] if train_folders else '.',
		'help':     "Location to search for model and save results.\nMust use location/<run_name>/model.pt structure.",
		'type':     str,
	},
	'agent': {
		'default':  'AStar',
		'help':     'Type of solver agent corresponding to agent class in librubiks.solving.agents',
		'type':     str,
		'choices':  ['AStar', 'MCTS', 'PolicySearch', 'ValueSearch', 'EGVM', 'BFS', 'RandomDFS', ],
	},
	'scrambling': {
		'default':  100,
		'help':     'The scrambling depths at which to test the model. Can be given as a singel integer for one depth or\n'
					'Two space-seperated integers (given in string delimeters such as --eval_scrambling "10 25")',
		# Ugly way to define list of two numbers or single number input
		'type':     lambda args: [int(args.split()[0]), int(args.split()[1])] if len(args.split()) > 1 else [int(args), int(args)+1],
	},
	'games': {
		'default':  500,
		'help':     'Number of games to play in evaluation for each depth, for each agent.',
		'type':     int,
	},
	'max_time': {
		'default':  0,
		'help':     'Max searching time for agent. 0 for unlimited. Evaluation is terminated when either max_time or max_states is reached.',
		'type':     float,
	},
	'max_states': {
		'default':  175_000,
		'help':     'Max number of searched states for agent per configuration. 0 for unlimited. Evaluation is terminated when either max_time or max_states is reached.',
		'type':     lambda arg: int(float(arg)),
	},
	'use_best': {
		'default':  True,
		'help':     "Set to True to use model-best.pt instead of model.pt.",
		'type':     literal_eval,
		'choices':  [True, False],
	},
	'optimized_params' : {
		'default':  False,
		'help':     "Set to True to overwrite agent params with the ones in corresponding JSON created by hyper_optim, if it exists.",
		'type':     literal_eval,
		'choices':  [True, False],
	},
	'astar_lambda' : {
		'default':  0.2,
		'help':     'The A* search lambda parameter: How much to weight the distance from start to nodes in cost calculation',
		'type':     float,
	},
	'astar_expansions' : {
		'default':  100,
		'help':     'The A* expansions parameter: How many nodes to expand to at a time. Can be thought of as a batch size: Higher is much faster but lower should be a bit more precise.',
		'type':     int,
	},
	'mcts_c': {
		'default':  0.6,
		'help':     'Exploration parameter c for MCTS',
		'type':     float,
	},
	'mcts_graph_search': {
		'default':  False,
		'help':     'Whether or not graph search should be applied to MCTS to find the shortest path',
		'type':     literal_eval,
		'choices':  [True, False],
	},
	'policy_sample': {
		'default':  False,
		'help':     'Whether or not there should be sampled when using the PolicySearch agent',
		'type':     literal_eval,
		'choices':  [True, False],
	},
	'egvm_epsilon': {
		'default':  0.01,
		'help':     'Epsilon for epsilon greedy policy search in Epsilon Greedy Value Maximization',
		'type':     float,
	},
	'egvm_workers': {
		'default':  10,
		'help':     'Number of sequential workers in Epsilon Greedy Value Maximization',
		'type':     int,
	},
	'egvm_depth': {
		'default':  100,
		'help':     'Exploration depth for each iteration of Epsilon Greedy Value Maximization',
		'type':     int,
	},
}

if __name__ == "__main__":
	description = r"""

___________________________________________________________________
  /_/_/_/\  ______ _      ______ _   _______ _____ _   __ _____
 /_/_/_/\/\ | ___ \ |     | ___ \ | | | ___ \_   _| | / //  ___|
/_/_/_/\/\/\| |_/ / |     | |_/ / | | | |_/ / | | | |/ / \ `--.
\_\_\_\/\/\/|    /| |     |    /| | | | ___ \ | | |    \  `--. \
 \_\_\_\/\/ | |\ \| |____ | |\ \| |_| | |_/ /_| |_| |\  \/\__/ /
  \_\_\_\/  \_| \_\_____/ \_| \_|\___/\____/ \___/\_| \_/\____/
__________________________________________________________________
Evaluate Rubiks agents using config or CLI arguments. If no location
is given, data/local_train with newest name is used. If the location
contains multiple neural networks, the deep agents are evalued for
each of them.
"""
	set_seeds()

	parser = Parser(options, description=description, name='eval', description_last='Tue')
	run_settings = parser.parse()
	jobs = [EvalJob(**settings, in_subfolder=len(run_settings)>1) for settings in run_settings]

	for job in jobs:
		job.execute()
	EvalJob.plot_all_jobs(jobs, parser.save_location)

