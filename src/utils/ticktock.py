from time import time

def get_timestamp(for_file=False):
	# Returns a timestamp
	# File name friendly format if for_file
	if for_file:
		return "-".join(str(datetime.now()).split(".")[0].split(":")).replace(" ", "_")
	else:
		return str(datetime.now())

class TickTock:

	_start: float

	def tick(self):
		self._start = time()
		return self._start
	
	def tock(self, with_print = True):
		end = time()
		passed_time = end - self._start
		if with_print:
			print(f"{passed_time/1000:3f} ms")
		return passed_time

