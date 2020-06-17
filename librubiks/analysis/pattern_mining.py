from librubiks.utils import Logger
from librubiks import cube
from librubiks.solving.agents import Agent, AStar


def find_patterns(sequence_list, support):
	all_actions = set(action for action_sequence in sequence_list for action in action_sequence)
	print(all_actions)
	sequence_list = [''.join(action_sequence) for action_sequence in sequence_list]
	print(sequence_list)
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
	print(patterns)
	return patterns


def find_generalized_patterns(sequence_list, support):
	sequence_list = [''.join(action_sequence) for action_sequence in sequence_list]
	patterns = {}
	for sequence in sequence_list:
		N = len(sequence)
		seen_subsequences = []
		for i in range(N):
			for j in range(2, N+1):
				if i+j < N+1:
					# for each subsequence, generalize it
					subsequence = sequence[i:i+j]
					generalized_subsequence = []
					alphabet_count = 0
					n = len(subsequence)
					for k in range(n):
						# check if it is the first time we see this move
						if subsequence[k] not in subsequence[:k-j]:
							name = 65 + alphabet_count
							# check if it is a reversing move
							if subsequence[k].lower() in subsequence[:k-j].lower():
								idx = subsequence[:k-j].lower().index(subsequence[k].lower())
								name = ord(generalized_subsequence[idx]) + 32
							else:
								alphabet_count += 1
						else:
							idx = subsequence[:k - j].lower().index(subsequence[k].lower())
							name = ord(generalized_subsequence[idx])
						generalized_subsequence.append(chr(name))
					generalized_subsequence = ''.join(generalized_subsequence)
					if generalized_subsequence not in patterns:
						patterns[generalized_subsequence] = 1
						seen_subsequences.append(generalized_subsequence)
					elif generalized_subsequence not in seen_subsequences:
					#else:
						patterns[generalized_subsequence] += 1
						seen_subsequences.append(generalized_subsequence)
	patterns = {pattern: patterns[pattern]/len(sequence_list) for pattern in patterns if patterns[pattern]/len(sequence_list) >= support}
	patterns = {k: v for k, v in sorted(patterns.items(), key=lambda item: item[1], reverse=True)}
	return patterns



# if both ab and abc are patterns, then ignore ab
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
	sequences = list()
	for i in range(games):
		actions_taken = []
		state, _, _ = cube.scramble(scramble, True)
		won = agent.search(state, max_time, None)
		if not won: log(f"Game {i+1} was not won")
		else:
			for action_num in agent.action_queue:
				action_tup = cube.action_space[action_num]
				actions_taken.append(cube.action_names[action_tup[0]].lower() if action_tup[1] else cube.action_names[action_tup[0]])
			log(f'Actions taken: {actions_taken}')
			sequences.append(actions_taken)
	return sequences



if __name__ == "__main__":
	### Hyper parameters ###
	net_path, use_best = '../rubiks-models/somerolloutcompare/smallroll', False
	max_time = 30
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
	log("Found patterns:")
	log(find_generalized_patterns(actions, support))
