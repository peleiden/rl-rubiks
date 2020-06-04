import numpy as np

from librubiks import cube, get_is2024, set_is2024, store_repr, restore_repr
from librubiks.utils import Logger, TickTock

def _repstr():
	return "20x24" if get_is2024() else "6x8x6"

def _get_states(shape: tuple):
	shape = (*shape, *cube.shape()) if len(shape) > 1 else (1, *shape, *cube.shape())
	n, n_states = shape[0], shape[1]
	states = np.empty(shape, dtype=cube.dtype)
	states[0] = cube.repeat_state(cube.get_solved(), n_states)
	for i in range(1, len(states)):
		faces, dirs = np.random.randint(0, 6, n_states), np.random.randint(0, 2, n_states)
		states[i] = cube.multi_rotate(states[i-1], faces, dirs)
	return states

class CubeBench:

	def __init__(self, log: Logger, tt: TickTock):
		self.log = log
		self.tt = tt
	
	def rotate(self, n: int):
		self.log.section(f"Benchmarking {TickTock.thousand_seps(n)} single rotations, {_repstr()}")
		faces, dirs = np.random.randint(0, 6, n), np.random.randint(0, 2, n)
		state = cube.get_solved()
		pname = f"Single rotation, {_repstr()}"
		for f, d in zip(faces, dirs):
			self.tt.profile(pname)
			state = cube.rotate(state, f, d)
			self.tt.end_profile()
		self._log_method_results("Average rotation time", pname)
	
	def multi_rotate(self, n: int, n_states: int):
		self.log.section(f"Benchmarking {TickTock.thousand_seps(n)} multi rotations of "
						 f"{TickTock.thousand_seps(n_states)} states each, {_repstr()}")
		states = cube.repeat_state(cube.get_solved(), n_states)
		faces, dirs = np.random.randint(0, 6, (n, n_states)), np.random.randint(0, 2, (n, n_states))
		pname = f"{TickTock.thousand_seps(n_states)} rotations, {_repstr()}"
		for f, d in zip(faces, dirs):
			self.tt.profile(pname)
			states = cube.multi_rotate(states, f, d)
			self.tt.end_profile()
		self._log_method_results("Average rotation time", pname, n_states)
	
	def onehot(self, n: int):
		self.log.section(f"Benchmarking {TickTock.thousand_seps(n)} one-hot encodings, {_repstr()}")
		states = _get_states((n,))
		pname = f"One-hot encoding single state, {_repstr()}"
		for state in states.squeeze():
			self.tt.profile(pname)
			cube.as_oh(state)
			self.tt.end_profile()
		self._log_method_results("Average state encoding time", pname)
	
	def multi_onehot(self, n: int, n_states: int):
		self.log.section(f"Benchmarking {TickTock.thousand_seps(n)} one-hot encodings of "
				 		 f"{TickTock.thousand_seps(n_states)} states each, {_repstr()}")
		all_states = _get_states((n, n_states))
		pname = f"One-hot encoding {TickTock.thousand_seps(n_states)} states, {_repstr()}"
		for states in all_states:
			self.tt.profile(pname)
			cube.as_oh(states)
			self.tt.end_profile()
		self._log_method_results("Average state encoding time", pname, n_states)
	
	def check_solution(self, n: int):
		self.log.section(f"Benchmarking {TickTock.thousand_seps(n)} single solution checks, {_repstr()}")
		states = _get_states((n,))
		pname = f"Checking single solution, {_repstr()}"
		for state in states.squeeze():
			self.tt.profile(pname)
			cube.is_solved(state)
			self.tt.end_profile()
		self._log_method_results("Average solution check time", pname)
	
	def check_multi_solution(self, n: int, n_states: int):
		self.log.section(f"Benchmarking {TickTock.thousand_seps(n)} solution checks of "
						 f"{TickTock.thousand_seps(n_states)} states each")
		all_states = _get_states((n, n_states))
		pname = f"Solution checks for {TickTock.thousand_seps(n_states)} states, {_repstr()}"
		for states in all_states:
			self.tt.profile(pname)
			cube.multi_is_solved(states)
			self.tt.end_profile()
		self._log_method_results("Average solution check time", pname, n_states)
	
	def _log_method_results(self, description: str, pname: str, divider=1):
		threshold = 2
		removed = self.tt.profiles[pname].remove_outliers(threshold)
		self.log("\n".join([
			description + ": " + TickTock.stringify_time(self.tt.profiles[pname].mean() / divider, "mus"),
			"Mean: " + TickTock.stringify_time(self.tt.profiles[pname].mean(), "mus"),
			"Std.: " + TickTock.stringify_time(self.tt.profiles[pname].std(), "mus"),
			f"Removed {TickTock.thousand_seps(removed)} outliers with threshold {threshold} * mean",
		]))

def benchmark():
	log = Logger("data/local_analyses/benchmarks.log", "Benchmarks")
	tt = TickTock()
	cube_bench = CubeBench(log, tt)

	# Cube config variables
	cn = int(1e7)
	multi_op_size = int(1e4)  # Number of states used in multi operations

	store_repr()
	for repr_ in [True, False]:
		set_is2024(repr_)
		log.section(f"Benchmarking cube enviroment with {_repstr()} representation")
		tt.profile(f"Benchmarking cube environment, {_repstr()}")
		cube_bench.rotate(cn)
		cube_bench.multi_rotate(int(cn/multi_op_size), multi_op_size)
		cube_bench.onehot(cn)
		cube_bench.multi_onehot(int(cn/multi_op_size), multi_op_size)
		cube_bench.check_solution(cn)
		cube_bench.check_multi_solution(int(cn/multi_op_size), multi_op_size)
		tt.end_profile(f"Benchmarking cube environment, {_repstr()}")
	
	restore_repr()
	
	log.section("Benchmark runtime distribution")
	log(tt)


if __name__ == "__main__":
	benchmark()

