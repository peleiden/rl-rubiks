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

# TODO Fix taking forever to search sometimes

def analyse_mcts(searchers: int, time_limit: float=1, output=True):
	# state, _, _ = Cube.scramble(50)
	state = np.array([int(x) for x in "11 14  7 19  5  1 16 22 15 10  9  6  1  4 18 23  3 16 13 20".split()], dtype=Cube.dtype)
	# net = Model(ModelConfig()).to(gpu).eval()
	# net = Model.load("data/hpc-20-04-12").to(gpu).eval()
	net = Model.load("data/local_errornet").to(gpu).eval()
	searcher = MCTS(net)
	solved = searcher.search(state, time_limit, searchers)
	assert not solved
	if output:
		log(searcher.tt)
		log(f"Tree size after {time_limit} s: {TickTock.thousand_seps(len(searcher.states))}")
	return len(searcher.states)

def optimize_searchers():
	x = np.arange(1, 401)
	y = []
	for s in x:
		sizes = [analyse_mcts(s, .5, False) for _ in range(5)]
		y.append(np.mean(sizes))
		log(f"Tree size at {s} / {len(x)} searchers: {y[-1]}")
	plt.plot(x, y)
	plt.grid(True)
	# plt.show()
	plt.savefig("data/local_analyses/mcts_searchers.png")
	
if __name__ == "__main__":
	# set_repr(False)
	seedsetter()
	analyse_mcts(100)
	# optimize_searchers()
	



