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
	plt.savefig("data/local_analyses/mcts_time_limit.png")
	# plt.show()
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
	plt.savefig("data/local_analyses/mcts_searchers.png")
	# plt.show()
	plt.clf()

def solve(depth: int, c: float, nu: float, workers: int, time_limit: float):
	state, _, _ = Cube.scramble(depth, True)
	net = Model.load("data/local_good_net").eval().to(gpu)
	searcher = MCTS(net, c, nu, False, workers)
	solved = searcher.search(state, time_limit)
	assert solved == (Cube.get_solved().tostring() in searcher.states)
	return solved, len(searcher.states)

def analyse_workers(n: int):
	workers = np.arange(1, 101, 20)
	y = []
	tree_sizes = []
	log.section(f"Optimizing number of workers\nExpected runtime: {len(workers)*.5*n} s")
	for x in workers:
		solved, lens = zip(*[solve(8, c=1, nu=.1, workers=x, time_limit=.5) for _ in range(n)])
		y.append(np.mean(solved))
		tree_sizes.append(np.mean(lens))
		log(f"Pct. solved at {x} workers: {y[-1]*100:.2f} %. Avg tree size: {tree_sizes[-1]:.0f}")
	fig, ax1 = plt.subplots()
	colour = "tab:blue"
	ax1.set_xlabel("Workers")
	ax1.set_ylabel("Share of cubes solved", color=colour)
	ax1.set_ylim([-.05, 1.05])
	ax1.plot(workers, y, color=colour)
	ax1.tick_params(axis="y", labelcolor=colour)
	ax1.grid(True)
	
	ax2 = ax1.twinx()
	colour = "tab:red"
	ax2.set_ylabel("Avg tree size")
	ax2.set_ylim(np.array([-.05, 1.05]))
	ax2.plot(workers, tree_sizes, color=colour)
	ax2.tick_params(axis="y", labelcolor=colour)
	
	# fig.grid(True)
	fig.savefig("data/local_analyses/mcts_workers.png")
	fig.show()
	fig.clf()
	

if __name__ == "__main__":
	# set_repr(False)
	n = 1
	# seedsetter()
	# analyse_mcts(100, 1)
	# optimize_time_limit()
	# optimize_searchers(n)
	analyse_workers(n)
	
	



