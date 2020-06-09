import functools
import numpy as np
import torch

cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rc_params = { "font.size": 22, "legend.fontsize": 18 }  # For matplotlib
rc_params_small = { "font.size": 18, "legend.fontsize": 16 }

def reset_cuda():
	torch.cuda.empty_cache()
	if torch.cuda.is_available(): torch.cuda.synchronize()


def no_grad(fun):
	functools.wraps(fun)
	def wrapper(*args, **kwargs):
		with torch.no_grad():
			return fun(*args, **kwargs)
	return wrapper


def softmax(x: np.ndarray, axis=0):
	e = np.exp(x)
	return (e.T / e.sum(axis=axis)).T if axis else e / e.sum(axis=axis)
