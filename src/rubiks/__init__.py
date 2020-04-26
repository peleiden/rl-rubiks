import torch

cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_cuda():
	torch.cuda.empty_cache()
	if torch.cuda.is_available(): torch.cuda.synchronize()

_is2024 = True
_stored_repr: bool

def set_is2024(is2024: bool):
	global _is2024
	assert type(is2024) is bool
	_is2024 = is2024

def get_is2024():
	return _is2024

def store_repr():
	global _stored_repr
	_stored_repr = _is2024

def restore_repr():
	global _is2024
	_is2024 = _stored_repr

def no_grad(fun):
	def wrapper(*args, **kwargs):
		with torch.no_grad():
			return fun(*args, **kwargs)
	return wrapper

