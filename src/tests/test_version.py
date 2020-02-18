import platform
import torch

def test_torch_version():
	assert torch.__version__ == '1.4.0'

def test_python_version():
	assert platform.architecture()[0] == "64bit"



