_is2024 = True

def set_repr(is2024: bool):
	global _is2024
	assert type(is2024) is bool
	_is2024 = is2024

def get_repr():
	global _is2024
	return _is2024
