import os

import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 22})

from src.rubiks import set_is2024
from src.rubiks.cube.cube import Cube
from src.rubiks.utils.ticktock import TickTock
from src.rubiks.utils.logger import Logger

import numpy as np

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

def sequence_sramble():
	n = 100
	tt.profile(f"Generating {n} sequence scrambles of depth {depth}")
	states, oh = Cube.sequence_scrambler(n, depth)
	tt.end_profile(f"Generating {n} sequence scrambles of depth {depth}")
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

	log.section("Sequence scrambles")
	for i in range(100):
		states = sequence_sramble()

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


if __name__ == "__main__":

	log.section("20x24 representation")
	analyse_cube()
	tt.reset()
	set_is2024(False)
	log.section("6x8x6 representation")
	analyse_cube()



