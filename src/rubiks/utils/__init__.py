import torch
import numpy as np
import random
from datetime import datetime

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
	if has_git:
		repo = git.Repo(".")  # TODO: Search upwards in directories
		return str(repo.head.commit)
	return "Unknown (install GitPython to get this)"



def get_timestamp(for_file=False):
	# Returns a timestamp
	# File name friendly format if for_file
	if for_file:
		return "-".join(str(datetime.now()).split(".")[0].split(":")).replace(" ", "_")
	else:
		return str(datetime.now())
