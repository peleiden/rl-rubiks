from datetime import datetime
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
	_sections = {}
	_units = {"ms": 1000, "s": 1, "m": 1/60}

	def tick(self):
		self._start = time()
		return self._start
	
	def tock(self, with_print = False):
		end = time()
		passed_time = end - self._start
		return passed_time

	def section(self, name: str):
		if name not in self._sections:
			self._sections[name] = {"tt": TickTock(), "elapsed": 0}
		self._sections[name]["tt"].tick()
	
	def end_section(self, name: str):
		dt = self._sections[name]["tt"].tock()
		self._sections[name]["elapsed"] += dt
	
	def get_sections(self):
		return {kw: v["elapsed"] for kw, v in self._sections.items()}
	
	@classmethod
	def stringify_time(cls, dt: float, unit: str):
		return f"{dt*cls._units[unit]:.3f} {unit}"
	
	def stringify_sections(self, unit="ms"):
		# Returns pretty sections
		sections = {kw: self.stringify_time(v, unit) for kw, v in self.get_sections().items()}
		strs = []
		for kw, v in sections.items():
			strs.append(f"{kw}: {v}")
		return "\n".join(strs)
	
	def __str__(self):
		return self.stringify_sections()