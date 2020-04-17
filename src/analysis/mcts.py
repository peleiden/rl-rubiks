import matplotlib.pyplot as plt
import numpy as np

from src.rubiks import gpu, set_repr
from src.rubiks.cube.cube import Cube
from src.rubiks.model import Model, ModelConfig
from src.rubiks.solving.search import Searcher, MCTS
from src.rubiks.utils import seedsetter
from src.rubiks.utils.logger import Logger
from src.rubiks.utils.ticktock import TickTock

tt = TickTock()
log = Logger("data/local_analyses/mcts.log", "Analyzing MCTS")


def solve(depth: int, c: float, nu: float, workers: int, time_limit: float):
	state, _, _ = Cube.scramble(depth, True)
	net = Model.load("data/local_good_net").eval().to(gpu)
	searcher = MCTS(net, c, nu, False, workers)
	solved = searcher.search(state, time_limit)
	assert solved == (Cube.get_solved().tostring() in searcher.states)
	return solved, len(searcher.states)

def analyze_var(var: str, values: np.ndarray, other_vars: dict):
	x = values
	y = []
	tree_sizes = []
	log.section(f"Optimizing {var}\nExpected runtime: {len(x)*time_limit*n:.2f} s\nGames per number of workers: {n}")
	log(f"Config\nTime limit per game: {time_limit:.2f} s\n{other_vars}")
	for val in values:
		vals = {**other_vars, var: val}
		solved, lens = zip(*[solve(**vals, time_limit=time_limit) for _ in range(n)])
		y.append(np.mean(solved))
		tree_sizes.append(max(lens))
		log(f"Pct. solved at {var} = {val:.4f}: {y[-1] * 100:.2f} %. Largest tree size: {tree_sizes[-1]:.0f}")
	fig, ax1 = plt.subplots()
	colour = "tab:blue"
	ax1.set_xlabel(var)
	ax1.set_ylabel("Share of cubes solved", color=colour)
	ax1.set_ylim([-.05, 1.05])
	ax1.plot(x, y, color=colour)
	ax1.tick_params(axis="y", labelcolor=colour)
	
	ax2 = ax1.twinx()
	colour = "tab:red"
	ax2.set_ylabel("Largest tree size")
	ax2.set_ylim(np.array([-.05, 1.05])*max(tree_sizes))
	ax2.plot(x, tree_sizes, color=colour)
	ax2.tick_params(axis="y", labelcolor=colour)
	
	fig.tight_layout()
	plt.title(f"Solving in {time_limit:.2f} s with {other_vars}")
	plt.grid(True)
	plt.savefig(f"data/local_analyses/mcts_{var}.png")
	plt.show()
	plt.clf()

if __name__ == "__main__":
	# set_repr(False)
	time_limit = .2
	n = 200
	default_vars = { "depth": 8, "c": 1, "nu": 0.01, "workers": 10 }
	get_other_vars = lambda excl: {kw: v for kw, v in default_vars.items() if kw != excl}
	# seedsetter()
	analyze_var(var="nu", values=np.linspace(0, 0.2, 30), other_vars=get_other_vars("nu"))




