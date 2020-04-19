import os

import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 22})
import numpy as np
from time import perf_counter

from src.rubiks import set_repr
from src.rubiks.cube.cube import Cube
from src.rubiks.utils.ticktock import TickTock
from src.rubiks.utils.logger import Logger

from datetime import datetime
from time import perf_counter
import numpy as np

# set_repr(False)
scrambles, depth, actions = 1000, 50, 100_000

tt = TickTock()
log = Logger(os.path.join("data", "local_analyses", "cube.log"), "Cube")

def scramble():
	states = np.empty((scrambles, *Cube.get_solved_instance().shape), dtype=Cube.dtype)
	for i in range(scrambles):
		tt.profile(f"Scrambling of depth {depth}")
		states[i], _, _ = Cube.scramble(depth)
		tt.end_profile(f"Scrambling of depth {depth}")
	return states

def oh(states):
	n_states = states.shape[-1]
	tt.profile(f"One-hot encoding of {n_states} states")
	oh = Cube.as_oh(states)
	tt.end_profile(f"One-hot encoding of {n_states} states")
	return oh

def perform_actions():
	state = Cube.get_solved_instance()
	for i in range(actions):
		action = Cube.action_space[np.random.choice(Cube.action_dim)]
		tt.profile("Performing one action")
		state = Cube.rotate(state, *action)
		tt.end_profile("Performing one action")

def analyse_cube():
	log("\n".join([
		"Analysing the cube environment",
		f"Number of scrambles: {scrambles}",
		f"Scrambling depth: {depth}",
		f"Actions: {TickTock.thousand_seps(actions)}",
	]))
	log.section("Scrambling")
	states = scramble()

	log.section(f"One-hot encoding {scrambles} states")
	for i in range(1000):
		oh(states)

	log.section(f"One-hot encoding one state {scrambles} times")
	for i in range(10_000):
		oh(Cube.get_solved())

	log.section(f"Performing {actions} actions")
	perform_actions()

	log.section("Running time")
	log(tt)

	# for kw, v in tt.get_sections().items():
	# 	print(kw)
	# 	print(f't[0] / max(t[1:]): {v["hits"][0] / max(v["hits"][1:]):.2f}')
	# 	plt.hist(v["hits"][1:], label=kw)
	# 	plt.title(kw)
	# 	plt.show()


if __name__ == "__main__":

	analyse_cube()



