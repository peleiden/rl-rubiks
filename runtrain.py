import os
from shutil import rmtree
from ast import literal_eval

from librubiks.utils import get_timestamp, Parser, seedsetter
from librubiks.jobs import TrainJob

####
# Should correspond to  arguments in librubiks.jobs.Trainjob
####
options = {
	'location': {
		'default':  'data/local_train'+get_timestamp(for_file=True),
		'help':     "Save location for logs and plots",
		'type':     str,
	},
	'rollouts': {
		'default':  500,
		'help':     'Number of passes of ADI+parameter update',
		'type':     int,
	},
	'rollout_games': {
		'default':  100,
		'help':     'Number of games in ADI in each rollout',
		'type':     int,
	},
	'rollout_depth': {
		'default':  50,
		'help':     'Number of scramblings applied to each game in ADI',
		"type":     int,
	},
	'batch_size': {
		'default':  50,
		'help':     'Number of training examples to be used in each parameter update',
		'type':     int
	},
	'alpha_update': {
		'default':  0,
		'help':     'alpha is set to alpha + alpha_update 100 times during training, though never more than 1. 0 for weighted and 1 for unweighted',
		'type':     float,
	},
	'lr': {
		'default':  1e-5,
		'help':     'Learning rate of parameter update',
		'type':     float,
	},
	'gamma': {
		'default':  1,
		'help':     'Learning rate reduction parameter. Learning rate is set updated as lr <- gamma * lr 100 times during training',
		'type':     float,
	},
	'tau': {
		'default':  1,
		'help':     'Network change parameter for generating training data. If tau=1, use newest network for ADI ',
		'type':     float,
	},
	'update_interval': {
		'default':  50,
		'help':     'How often alpha and lr are updated. First update is performed when rollout == update_interval. Set to 0 for never',
		'type':     int,
	},
	'reward_method' : {
		'default':  'lapanfix',
		'help':     'Which way to set target values near goal state. "paper" forces nothing and does not train on goal state. ' +
		            '"lapanfix" trains on goalstate and forces it = 0. "schultzfix" forces substates for goal to 0 and does not train on goal state. ' +
		            '"reward0" changes reward for transitioning to goal state to 0 and does not train on goal state.',
		'type':     str,
		'choices':  ['paper', 'lapanfix', 'schultzfix', 'reward0'],
	},
	'optim_fn': {
		'default':  'RMSprop',
		'help':     'Name of optimization function corresponding to class in torch.optim',
		'type':     str,
	},
	'evaluation_interval': {
		'default':  100,
		'help':     'An evaluation is performed every evaluation_interval rollouts. Set to 0 for never',
		'type':     int,
	},
	'nn_init': {
		'default':  'glorot',
		'help':     'Initialialization strategy for the NN. Choose either "glorot", "he" or write a number. If a number is given, the network is initialized to this constant.',
		'type':     str,
	},
	'is2024': {
		'default':  True,
		'help':     'True for 20x24 Rubiks representation and False for 6x8x6',
		'type':     literal_eval,
		'choices':  [True, False],
	},
	'arch': {
		'default':  'fc',
		'help':     'Network architecture. fc for fully connected, res for fully connected with residual blocks, and conv for convolutional blocks',
		'type':     str,
		'choices':  ['fc', 'res', 'conv'],
	},
	'analysis': {
		'default':  False,
		'help':    'If true, analysis of model changes, value and loss behaviour is done in each rollout and ADI pass',
		'type':     literal_eval,
		'choices':  [True, False],
	},
}

def clean_dir(loc: str):
	"""
	Cleans a training directory created by runtrain
	All except the config file is removed
	"""
	with open(f"{loc}/train_config.ini") as f:
		content = f.read()
	rmtree(loc)
	os.mkdir(loc)
	return content

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
	cfg_content = clean_dir(parser.save_location)
	for job in jobs:
		job.execute()
	with open(f"{parser.save_location}/train_config.ini", "w") as f:
		f.write(cfg_content)
