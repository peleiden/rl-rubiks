import functools
import numpy as np
import torch

cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rc_params = { "font.size": 24, "legend.fontsize": 22, "legend.framealpha": 0.5 }  # matplotlib settings
rc_params_small = { **rc_params, "font.size": 20, "legend.fontsize": 18 }  # Same but with smaller font

def reset_cuda():
	torch.cuda.empty_cache()
	if torch.cuda.is_available(): torch.cuda.synchronize()


def no_grad(fun):
	functools.wraps(fun)
	def wrapper(*args, **kwargs):
		with torch.no_grad():
			return fun(*args, **kwargs)
	return wrapper

