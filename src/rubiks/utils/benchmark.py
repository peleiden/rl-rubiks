from os import makedirs
from shutil import rmtree
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from utils.ticktock import TickTock
from utils.logger import Logger

class Benchmark:

	def __init__(self, fun, outdir: str, title: str = ""):
		rmtree(f"{outdir}/", ignore_errors=True)
		makedirs(outdir)
		self.title = title if title else outdir
		self.tt = TickTock()
		self.log = Logger(f"{outdir}/benchmark.log", title)
		self.fun = fun
	
	def singlethreaded(self, data_desc = "", *args, **kwargs):
		self.log(f"Beginning single threaded benchmark of function {self.fun.__name__}")
		if data_desc:
			self.log(f"Data description:\n{data_desc}\n")
		try:
			self.tt.tick()
			self.fun(*args, **kwargs)
			t = self.tt.tock()
			self.log(f"Benchmark finished in time\n{t:f} s\n")
			return t
		except Exception as e:
			self.log(f"Test crashed with exception\n{e}\n")
			return 0
	
	def multithreaded(self, threads: range, data: list, data_desc = ""):
		threads = np.array(threads)
		times = np.empty(len(threads))
		self.log(f"Beginning multi threaded benchmark of function {self.fun.__name__}")
		if data_desc:
			self.log(f"Data description:\n{data_desc}\n")
		self.log("Benchmark results")
		try:
			for i, n_treads in enumerate(threads):
				with mp.Pool(n_treads) as p:
					self.tt.tick()
					p.map(self.fun, data)
					times[i] = self.tt.tock()
					self.log(f"{n_treads} threads: {times[i]:f} s", with_timestamp=False)
			self.log("")
			return threads, times
		except Exception as e:
			self.log(f"Test crashed with exception\n{e}\n")
			return [], []
	
	def plot_mt_results(self, threads, times, title: str = ""):
		plt.plot(threads, times)
		plt.xlabel("Number of threads")
		plt.ylabel("Runtime [s]")
		if title:
			plt.title(title)
		plt.grid(True)
		plt.show()




