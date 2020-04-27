import torch
import numpy as np
import random
try:
	import git
	has_git = True
except ModuleNotFoundError:
	has_git = False

def seedsetter():
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(0)
	random.seed(0)

def get_commit():
	repo = git.Repo(".")
	return str(repo.head.commit) if has_git else "Unknown (install GitPython to get this)"

