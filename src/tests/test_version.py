import platform
import torch
import scipy

def test_torch_version():
	v = torch.__version__.split(".")
	assert int(v[0]) == 1 and int(v[1]) >= 4

def test_python_version():
	assert platform.architecture()[0] == "64bit"

def test_scipy_version():
	# Needed for Shannons entropy
	v = scipy.__version__.split(".")
	assert int(v[0]) == 1 and int(v[1]) >= 4

