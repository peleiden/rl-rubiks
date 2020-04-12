from src.rubiks.utils.ticktock import TickTock

import numpy as np
from time import sleep

def test_tt():
	tt = TickTock()
	tt.section("test0")
	sleep(.01)
	tt.section("test1")
	sleep(.01)
	tt.end_section("test1")
	sleep(.01)
	tt.end_section("test0")
	secs = tt.get_section_times()
	assert np.isclose(0.03, secs["test0"], .1)
	assert np.isclose(0.01, secs["test1"], .1)
	