import torch
import numpy as np
import random

cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seedsetter():
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(0)
	random.seed(0)


