import numpy as np
from time import sleep

from librubiks.utils import TickTock

def test_tt():
	tt = TickTock()
	tt.profile("test0")
	sleep(.01)
	tt.profile("test1")
	sleep(.01)
	tt.end_profile("test1")
	sleep(.01)
	tt.end_profile("test0")
	assert np.isclose(0.03, tt.profiles["test0"].sum(), 1)
	assert np.isclose(0.01, tt.profiles["test1"].sum(), 1)

