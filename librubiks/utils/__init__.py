import torch
import numpy as np
import random
from datetime import datetime
from scipy.stats import norm

try:
	import git
	has_git = True
except ModuleNotFoundError:
	has_git = False

def set_seeds():
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(0)
	random.seed(0)


_quick_zs = {
	0.1  : 1.6448536269514722,
	0.05 : 1.959963984540054,
	0.01 : 2.5758293035489004,
}
def bernoulli_error(p: float, n: int, alpha: float, stringify: bool=False):
	try: z = _quick_zs[alpha]
	except KeyError: z = norm.ppf(1-alpha/2)
	error = z * np.sqrt(p * (1-p) / n )
	if stringify: return f"+/- {error*100:.0f} %"
	return  error

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


# To allow imports directly from utils #
# Must be placed lower
from .logger import *
from .parse import *
from .ticktock import *
