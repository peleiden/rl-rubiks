import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 22})
import numpy as np
np.set_printoptions(precision=4, threshold=np.inf)

from librubiks import gpu, set_is2024
from librubiks import cube
from librubiks.model import Model, ModelConfig
from librubiks.solving.agents import Agent, MCTS, PolicySearch, BFS, ValueSearch

from librubiks.utils import seedsetter, Logger, TickTock

tt = TickTock()
log = Logger("data/local_analyses/mcts.log", "Analyzing MCTS")
net = Model.load("data/local_method_comparison/asgerfix").eval().to(gpu)


def solve(depth: int, c: float, nu: float, workers: int, time_limit: float, policy_type: str):
	state, f, d = cube.scramble(depth, True)
	searcher = MCTS(net, c=c, search_graph=False)
	is_solved = searcher.search(state, time_limit)
	assert is_solved == (cube.get_solved().tostring() in searcher.indices)
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

def analyse_time_distribution(depth: int, c: float):
	time_limits = np.linspace(.1, 2, 10)
	expand = np.zeros_like(time_limits)
	explore = np.zeros_like(time_limits)
	searcher = MCTS(net, c=c, search_graph=False)
	log.section(f"Analyzing time distribution at depth {depth}\nExpected max time <~ {TickTock.stringify_time(sum(time_limits*n), 'm')}")
	for i, tl in enumerate(time_limits):
		log(f"Analyzing with time limit of {tl:.2f} s")
		sols = np.zeros(n)
		for j in range(n):
			state, f, d = cube.scramble(depth, True)
			sols[j] = searcher.search(state, time_limit=tl)
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
	plt.savefig(f"data/local_analyses/mcts_time.png")
	# plt.show()
	plt.clf()

def detailed_time(state, searcher, max_states: int, time_limit: float, c: float):
	searcher = searcher(Model.load("data/local_train"), c=c, search_graph=False)
	log.section(f"Detailed time analysis: {searcher}")
	sol_found = searcher.search(state, time_limit, max_states)
	log("Solved found" if sol_found else "Solved not found")
	log(f"States explored: {len(searcher)}")
	log(searcher.tt)

def W(max_states, time_limit, opts):
	# state = np.array("8 18 12  2  5 17 22 11 14 22 12 11 17  2  7  4 21  9 19  0".split(), dtype=cube.dtype)
	# state = np.array("11  3 23  6  2 15 18 14  6  2  4 15  1 10 12 17  9 18 20 22".split(), dtype=cube.dtype)
	state, f, d = cube.scramble(50)
	log("State:", state)
	searcher = MCTS.from_saved("data/local_method_comparison/asgerfix", use_best=False, search_graph=False, **opts)
	log.section("Analyzing W")
	log("Solved", searcher.search(state, time_limit=time_limit, max_states=max_states))
	log(f"Number of states: {len(searcher)}")
	log(f"Share of W = 0: {np.mean(searcher.W[1:len(searcher)+1]==0):.2f}")
	log("Share of L = 0", np.mean(searcher.L==0))
	log("Share of all N = 0", np.mean((searcher.N==0).all(axis=1)))
	best_index = searcher.V[1:len(searcher)+1].argmax() + 1
	searcher._complete_graph()
	searcher._shorten_action_queue(best_index)
	log("Best value", searcher.V[best_index], "at index", best_index)
	log("State at best index", searcher.states[best_index])
	# log(cube.stringify(agent.states[best_index]))
	
	print(searcher.tt)
	
	samples = np.arange(1, len(searcher)+1, )#len(agent)//1000)
	plt.figure(figsize=(15, 15))
	plt.subplot(311)
	plt.scatter(samples, searcher.V[samples], s=2, label="V")
	plt.scatter(samples, searcher.W[samples].mean(axis=1), s=2, label="W")
	plt.legend(loc=4)
	plt.grid(True)
	
	plt.subplot(312)
	assert np.all(np.isclose(searcher.P[1:len(searcher)+1].sum(axis=1), 1, atol=1e-5))
	p = searcher.P.max(axis=1)
	n = searcher.N.max(axis=1)/searcher.N.sum(axis=1)
	n[np.isnan(n)] = 0
	plt.scatter(samples, p[samples], s=2,  label="P")
	plt.scatter(samples, n[samples], s=3, label="N")
	plt.legend(loc=4)
	plt.grid(True)
	
	plt.subplot(313)
	states = [cube.get_solved()]
	for f_, d_ in zip(f, d):
		states.append(cube.rotate(states[-1], f_, d_))
	for action in searcher.action_queue:
		states.append(cube.rotate(states[-1], *cube.action_space[action]))
	states = np.array(states, dtype=cube.dtype)
	oh_states = cube.as_oh(states)
	values = net(oh_states, policy=False).detach().cpu().squeeze().numpy()
	assert np.isclose(values[-1], searcher.V[best_index], atol=1e-5)
	plt.plot(values)
	plt.axvline(len(f))
	plt.grid(True)
	
	plt.savefig("data/local_analyses/mcts_W.png")
	plt.clf()


if __name__ == "__main__":
	# set_repr(False)
	time_limit = 5
	n = 100
	default_vars = { "depth": 15, "c": 5 }
	get_other_vars = lambda excl: {kw: v for kw, v in default_vars.items() if kw != excl}
	# seedsetter()
	# analyze_var(var="nu", values=np.linspace(0, 0.06, 30), other_vars=get_other_vars("nu"))
	# analyze_var(var="depth", values=np.arange(1, 21, 1), other_vars=get_other_vars("depth"))
	# analyze_var(var="c", values=np.logspace(-2, 2, 10), other_vars=get_other_vars("c"))
	# analyze_var(var="workers", values=np.unique(np.logspace(0, 1.7, 30).astype(int)), other_vars=get_other_vars("workers"))
	n = 40
	#analyse_time_distribution(25, 0.5, 0.005, 10, "p")
	#analyse_time_distribution(25, 0.5, 0.005, 100, "p")
	s = int(1e6)
	tl = 1
	# state, _, _ = cube.scramble(50)
	# detailed_time(state, MCTS, s, tl, 0.6, 0.005, 10, "p")
	W(None, 10, get_other_vars("depth"))

