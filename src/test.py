from utils.ticktock import TickTock
import numpy as np
from rubiks import RubiksCube
import torch

cpu = torch.device("cpu")
gpu = torch.device("cuda")

def to_cuda(a):
	return torch.from_numpy(a).cuda()

def to_cpu(a):
	return a.cpu().numpy()

if __name__ == "__main__":

	rube = RubiksCube()
	tt = TickTock()
	tt.tick()
	a = to_cuda(rube.state)
	tt.tock()
	tt.tick()
	b = to_cpu(a)
	tt.tock()
