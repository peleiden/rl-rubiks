import os

import numpy as np

from src.rubiks.cube.cube import Cube
from src.rubiks.utils.ticktock import TickTock
from src.rubiks.utils.logger import Logger

def scramble():
	states = np.empty((scrambles, *Cube.get_solved_instance().shape), dtype=Cube.dtype)
	for i in range(scrambles):
		tt.section(f"Scrambling of depth {depth}")
		states[i], _, _ = Cube.scramble(depth)
		tt.end_section(f"Scrambling of depth {depth}")
	return states

def oh(states):
	n_states = 1 if len(states.shape) == 1 else len(states)
	tt.section(f"One-hot encoding of {n_states} states")
	oh = Cube.as_oh(states)
	tt.end_section(f"One-hot encoding of {n_states} states")
	return Cube.as_oh(states)

def perform_actions():
	state = Cube.get_solved_instance()
	for i in range(actions):
		action = Cube.action_space[np.random.choice(Cube.action_dim)]
		tt.section("Performing one action")
		state = Cube.rotate(state, *action)
		tt.end_section("Performing one action")

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


if __name__ == "__main__":
	scrambles, depth, actions = 1000, 50, 100_000

	tt = TickTock()
	log = Logger(os.path.join("data", "local_analyses", "cube.log"), "Cube")
	analyse_cube()



