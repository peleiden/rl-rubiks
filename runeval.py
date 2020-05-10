from glob import glob as glob #glob

from ast import literal_eval

from librubiks.utils import Parser, seedsetter
from librubiks.jobs import EvalJob

train_folders = sorted(glob('data/local_train2*')) #Stops working in the next millenium

###
# Should correspond to arguments in EvalJob
###
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
		'help':	    'Max searching time for agent',
		'type':	    float,
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

