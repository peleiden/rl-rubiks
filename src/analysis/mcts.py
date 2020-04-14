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
	state, _, _ = Cube.scramble(50)
	net = Model(ModelConfig()).to(gpu).eval()
	# net = Model.load("data/hpc-20-04-12").to(gpu).eval()
	searcher = MCTS(net, c=1, nu=1)
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
	analyse_mcts(50)
	optimize_searchers()
	



