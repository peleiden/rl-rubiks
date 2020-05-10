import matplotlib.pyplot as plt
import numpy as np

from librubiks.cube import Cube
from librubiks.utils import TickTock

def estimate_state_space(d: int):
	"""
	Estimates the size of the state space at depth d
	This is a rough estimation will multiple doubtful assumptions
	Some probability theory would do nicely here
	"""
	n = int(1e7)
	states = np.array([Cube.get_solved() for _ in range(n)], dtype=Cube.dtype)
	for step in range(d):
		faces, dirs = np.random.randint(0, 6, n), np.random.randint(0, 2, n)
		states = Cube.multi_rotate(states, faces, dirs)
	uniques = {x.tostring() for x in states}

	m = int(n / 10)
	states = np.array([Cube.get_solved() for _ in range(m)], dtype=Cube.dtype)
	for step in range(d):
		faces, dirs = np.random.randint(0, 6, m), np.random.randint(0, 2, m)
		states = Cube.multi_rotate(states, faces, dirs)
	observed = [x.tostring() in uniques for x in states]

	obs_share = np.mean(observed)
	estim_state_space = int(len(uniques) / obs_share) if obs_share else 1
	ub = 12 * 11 ** (d-1) if d else 1  # Upper bound on state space size
	return estim_state_space, ub


if __name__ == "__main__":
	sizes = []
	depths = np.arange(30)
	for d in depths:
		s, ub = estimate_state_space(d)
		print(f"Estimated state space size at depth {d}: {TickTock.thousand_seps(s)} and upper bound {TickTock.thousand_seps(ub)}")
		sizes.append(s)
	plt.plot(depths, sizes, label="Estimated size of state space")
	plt.xlabel("Depth")
	plt.ylabel("Estimated size of state space")
	plt.axhline(4.3e19, color="orange", label="Total size of state space")
	plt.semilogy()
	plt.grid(True)
	plt.show()

