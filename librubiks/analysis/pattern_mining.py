from librubiks.utils import Logger
from librubiks import cube
from librubiks.solving.agents import Agent, AStar


def find_patterns(sequence_list, support):
	all_actions = set(action for action_sequence in sequence_list for action in action_sequence)
	sequence_list = [''.join(action_sequence) for action_sequence in sequence_list]
	patterns = [('', 0)]
	for start in patterns:
		for end in all_actions:
			count = 0
			action = ''.join([start[0], end])
			for action_sequence in sequence_list:
				if action in action_sequence: count += 1
			action_support = count/len(sequence_list)
			if action_support >= support:
				patterns.append((action, action_support))
	patterns = patterns[1:]
	return patterns


def trim_patterns(patterns):
	trimmed_patterns = []
	for i, sub_seq in enumerate(patterns):
		trim = False
		for j, super_seq in enumerate(patterns[i+1:]):
			if sub_seq[0] in super_seq[0]:
				trim = True
				break
		if not trim: trimmed_patterns.append(sub_seq)
	return trimmed_patterns


def sort_patterns(patterns):
	return sorted(patterns, key=lambda x: x[1], reverse=True)


def generate_actions(agent: Agent, games: int, max_time: float, scramble: int = 100):
	actions_taken = list()
	for i in range(games):
		state, _, _ = cube.scramble(scramble, True)
		won = agent.search(state, max_time, None)
		if not won: log(f"Game {i} was not won")
		for action_num in agent.action_queue:
			action_tup = cube.action_space[action_num]
			actions_taken.append(cube.action_names[action_tup[0]].lower() if action_tup[1] else cube.action_names[action_tup[0]])
	return actions_taken



if __name__ == "__main__":
	### Hyper parameters ###
	net_path, use_best = '../rubiks-models/somerolloutcompare/smallroll', False
	max_time = 10
	lambda_, N = 0.2, 10

	output_path = 'data/patterns.log'
	games = 10
	support = 0.4
	########################
	log = Logger(output_path, "Pattern mining")
	agent = AStar.from_saved(net_path, use_best, lambda_, N)
	log(f"Loaded agent {agent} with network {net_path}")

	log(f"Playing {games} games")
	actions = generate_actions(agent, games, max_time)
	log("Finding patterns...")
	patterns = find_patterns(actions, support)
	patterns = trim_patterns(patterns)
	log("Found patterns:")
	log(sort_patterns(patterns))

	# eksempel på løsninger af forskellige længder og handlinger
	# har flg mønstre: R (2x), r l (2x), l D (2x), D L (2x), l u d (3x)
	# test_set = [['R', 'r', 'l', 'u', 'd', 'D', 'L'],
			# ['r', 'R', 'l', 'u', 'd', 'L'],
			# ['l', 'D', 'D', 'L', 'l', 'u', 'd'],
			# ['r', 'r', 'r'],
			# ['r', 'l', 'D', 'l', 'L']]
