import platform
import torch
import scipy

def test_torch_version():
	assert torch.__version__.startswith("1.5")

def test_python_version():
	assert platform.architecture()[0] == "64bit"

def test_scipy_version():
	# Needed for Shannons entropy
	v = scipy.__version__.split(".")
	assert int(v[0]) == 1 and int(v[1]) >= 4

