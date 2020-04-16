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

def analyse_mcts(workers: int, time_limit: float=1, output=True):
	state, _, _ = Cube.scramble(50)
	net = Model(ModelConfig()).to(gpu).eval()
	searcher = MCTS(net, c=1, nu=10, search_graph=False)
	solved = searcher.search(state, time_limit, workers)
	assert not solved
	if output:
		log(searcher.tt)
		log(f"Tree size after {time_limit} s: {TickTock.thousand_seps(len(searcher.states))}")
	return len(searcher.states)

def optimize_time_limit():
	workers = 100
	runs = np.linspace(20, 2, 100).astype(int)
	time_limits = np.linspace(0.01, 1, 100)
	y = []
	log.section(f"Optimizing time limit with {workers} workers\nExpected runtime: {runs@time_limits} s")
	for s, r in zip(time_limits, runs):
		sizes = [analyse_mcts(workers, s, False) for _ in range(r)]
		y.append(np.mean(sizes)/s)
		log(f"Explored states per second at {s:.2f} s: {y[-1]:.2f}. Mean of {r} runs")
	plt.plot(time_limits, y)
	plt.xlabel("Time limit")
	plt.ylabel("Explored states per second")
	plt.grid(True)
	# plt.show()
	plt.savefig("data/local_analyses/mcts_time_limit.png")
	plt.clf()

def optimize_searchers(n: int):
	x = np.arange(1, 201)
	y = []
	log.section(f"Optimizing number of searchers\nExpected runtime: {len(x)*1*n} s")
	for s in x:
		sizes = [analyse_mcts(s, 1, False) for _ in range(n)]
		y.append(np.mean(sizes))
		log(f"Tree size at {s} / {len(x)} workers: {y[-1]}. Mean of {n} runs")
	plt.plot(x, y)
	plt.xlabel("Workers")
	plt.ylabel("Tree size with time limit of 1 s")
	plt.grid(True)
	# plt.show()
	plt.savefig("data/local_analyses/mcts_searchers.png")
	plt.clf()

if __name__ == "__main__":
	# set_repr(False)
	n = 5
	analyse_mcts(100, 1)
	optimize_time_limit()
	optimize_searchers(n)
	



