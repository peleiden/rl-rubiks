import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 22})
import numpy as np

from src.rubiks import gpu, set_is2024
from src.rubiks.cube.cube import Cube
from src.rubiks.model import Model, ModelConfig
from src.rubiks.solving.search import Searcher, MCTS, MCTS2
from src.rubiks.utils import seedsetter
from src.rubiks.utils.logger import Logger
from src.rubiks.utils.ticktock import TickTock

tt = TickTock()
log = Logger("data/local_analyses/mcts.log", "Analyzing MCTS")
net = Model.load("data/local_good_net").eval().to(gpu)


def solve(depth: int, c: float, nu: float, workers: int, time_limit: float):
	state, f, d = Cube.scramble(depth, True)
	searcher = MCTS(net, c, nu, False, False, workers)
	is_solved = searcher.search(state, time_limit, int(1e10))
	assert is_solved == (Cube.get_solved().tostring() in searcher.indices)
	return is_solved, len(searcher.indices)

def analyze_var(var: str, values: np.ndarray, other_vars: dict):
	x = values
	y = []
	tree_sizes = []
	log.section(f"Optimizing {var}\nExpected runtime: {len(x)*time_limit*n:.2f} s\nGames per evaluation: {n}")
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
	plt.title(f"Solving in {time_limit:.2f} s with {other_vars}. Mean of {n} games")
	plt.grid(True)
	plt.savefig(f"data/local_analyses/mcts_{var}.png")
	# plt.show()
	plt.clf()

def analyse_time_distribution(depth: int, c: float, nu: float, workers: int):
	time_limits = np.linspace(.1, 2, 10)
	expand = np.zeros_like(time_limits)
	explore = np.zeros_like(time_limits)
	searcher = MCTS(net, c=c, nu=nu, workers=workers, complete_graph=False, search_graph=False)
	log.section(f"Analyzing time distribution at depth {depth}\nExpected max time <~ {TickTock.stringify_time(sum(time_limits*n), 'm')}")
	for i, tl in enumerate(time_limits):
		log(f"Analyzing with time limit of {tl:.2f} s")
		sols = np.zeros(n)
		for j in range(n):
			state, f, d = Cube.scramble(depth, True)
			sols[j] = searcher.search(state, time_limit=tl, max_states=int(1e10))
			expand[i] += sum(searcher.tt.profiles["Expanding leaves"].hits)
			try:
				explore[i] += sum(searcher.tt.profiles["Exploring next node"].hits)
			except KeyError:
				pass
		log(f"Solved {np.mean(sols)*100:.2f} % of configurations")
	expand /= n
	explore /= n
	expand, explore = expand / (expand + explore), explore / (expand + explore)
	
	plt.figure(figsize=(15, 10))
	plt.plot(time_limits, expand*100, "o-", label="Time spent expanding")
	plt.plot(time_limits, explore*100, "o-", label="Time spent exploring")
	plt.legend(loc=2)
	plt.xlabel("Time limit [s]")
	plt.ylabel(f"Mean time spent over {n} runs [%]")
	plt.ylim([-0.05, 1.05])
	# plt.semilogx()
	plt.grid(True)
	plt.savefig(f"data/local_analyses/mcts_time_w={workers}.png")
	# plt.show()
	plt.clf()

def detailed_time(state, searcher, max_states: int, time_limit: float, c: float, nu: float, workers: int):
	searcher = searcher(Model.load("data/local_train"), c=c, nu=nu, complete_graph=False, search_graph=False, workers=workers)
	log.section(f"Detailed time analysis: {searcher}")
	sol_found = searcher.search(state, time_limit, max_states)
	log("Solved found" if sol_found else "Solved not found")
	log(f"States explored: {len(searcher)}")
	log(searcher.tt)

if __name__ == "__main__":
	# set_repr(False)
	time_limit = .2
	n = 200
	default_vars = { "depth": 8, "c": 1, "nu": 0.005, "workers": 10 }
	get_other_vars = lambda excl: {kw: v for kw, v in default_vars.items() if kw != excl}
	seedsetter()
	#analyze_var(var="nu", values=np.linspace(0, 0.06, 30), other_vars=get_other_vars("nu"))
	#analyze_var(var="depth", values=np.arange(1, 21, 1), other_vars=get_other_vars("depth"))
	#analyze_var(var="c", values=np.linspace(0, 20, 30), other_vars=get_other_vars("c"))
	#analyze_var(var="workers", values=np.unique(np.logspace(0, 1.7, 30).astype(int)), other_vars=get_other_vars("workers"))
	n = 40
	analyse_time_distribution(25, 0.5, 0.005, 10)
	analyse_time_distribution(25, 0.5, 0.005, 100)
	s = int(1e6)
	tl = 1
	state, _, _ = Cube.scramble(50)
	detailed_time(state, MCTS, s, tl, 0.6, 0.005, 10)
	detailed_time(state, MCTS2, s, tl, 0.6, 0.005, 10)




