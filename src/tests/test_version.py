import platform
import torch

def test_torch_version():
	assert "1.4.0" in torch.__version__

def test_python_version():
	assert platform.architecture()[0] == "64bit"



