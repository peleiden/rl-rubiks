from utils.ticktock import TickTock
import numpy as np
from rubiks import RubiksCube
import matplotlib.pyplot as plt
import torch
from pprint import pprint

cpu = torch.device("cpu")
gpu = torch.device("cuda")

def gen_data(size, dtype):
	return torch.randint(0, 1, size=(int(size),), dtype=dtype)

if __name__ == "__main__":
	sizes = torch.logspace(1, 5, 128)
	dtypes = {torch.int8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8}
	transfertimes = {str(dtype): [[], []] for dtype in dtypes}
	tt = TickTock()
	for dtype in dtypes:
		print(dtype)
		for size in sizes:
			nbytes = dtypes[dtype] * size
			dat = gen_data(size, dtype)
			dat.to(gpu)
			tt.tick()
			dat.to(gpu)
			time = tt.tock(False)
			transfertimes[str(dtype)][0].append(nbytes)
			transfertimes[str(dtype)][1].append(time)
	pprint(transfertimes)
	
	dtype = str(torch.int64)
	y = transfertimes[dtype][1]
	A = np.ones((len(sizes), 2))
	A[:, 1] = transfertimes[dtype][0]
	b, a = np.linalg.solve(A.T @ A, A.T @ y)
	print("a, b =", a, b)

	for dtype, time in transfertimes.items():
		plt.plot(time[0], time[1], label=dtype)
		plt.legend(loc=2)
	plt.xlabel("Number of bytes")
	plt.ylabel("Transfer time [s]")
	plt.grid(True)
	plt.show()


