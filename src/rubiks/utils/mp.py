import multiprocessing as mp
import numpy as np
CPUS = mp.cpu_count()

def multi_exec(fun, n: int, *args, **kwargs):
	"""
	Applies the function `fun` n times using multithreading
	"""
	# Determines how many workers to use
	workers = min(CPUS, n)
	
	def apply(_):
		return fun(*args, **kwargs)
	
	with mp.Pool(workers) as p:
		res = p.map(apply, [None]*n)
	
	return res

