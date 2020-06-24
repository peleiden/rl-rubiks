import numpy as np

from librubiks.utils import Logger
from librubiks import cube
from librubiks.solving.agents import Agent, AStar


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
						patterns[generalized_subsequence] += 1
						seen_subsequences.append(generalized_subsequence)
	patterns = {pattern: patterns[pattern]/len(sequence_list) for pattern in patterns if patterns[pattern]/len(sequence_list) >= support}
	patterns = {k: v for k, v in sorted(patterns.items(), key=lambda item: item[1], reverse=True)}
	return patterns


def generate_actions(agent: Agent, games: int, max_time: float):
	sequences = list()
	for i in range(games):
		actions_taken = []
		state, _, _ = cube.scramble(np.random.randint(100, 1000), True)
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
	net_path, use_best = '../rubiks-models/main', True
	max_time = 5
	lambda_, N = 0.16, 700

	output_path = '../rubiks-models/main/patterns.log'
	games = 1000
	support = 0.3

	########################
	log = Logger(output_path, "Pattern mining")
	agent = AStar.from_saved(net_path, use_best, lambda_, N)
	log(f"Loaded agent {agent} with network {net_path}")

	log(f"Playing {games} games")
	actions = generate_actions(agent, games, max_time)
	log("Found patterns:")
	log(find_generalized_patterns(actions, support))
