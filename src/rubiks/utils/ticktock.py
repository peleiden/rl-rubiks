from datetime import datetime
from time import perf_counter
import numpy as np

def get_timestamp(for_file=False):
	# Returns a timestamp
	# File name friendly format if for_file
	if for_file:
		return "-".join(str(datetime.now()).split(".")[0].split(":")).replace(" ", "_")
	else:
		return str(datetime.now())


class TickTock:

	_start: float = 0.
	_sections = {}
	_section_depth = 0
	_units = {"ns": 1e9, "mus": 1e6, "ms": 1e3, "s": 1, "m": 1/60}

	def tick(self):
		self._start = perf_counter()
		return self._start

	def tock(self):
		end = perf_counter()
		passed_time = end - self._start
		return passed_time

	def section(self, name: str):
		if name not in self._sections:
			self._sections[name] = {"tt": TickTock(), "hits": [], "depth": self._section_depth}
		self._section_depth += 1
		self._sections[name]["tt"].tick()

	def end_section(self, name: str):
		dt = self._sections[name]["tt"].tock()
		self._sections[name]["hits"].append(dt)
		self._section_depth -= 1
	
	def rename_section(self, name: str, new_name: str):
		# Renames a section
		# If a section with new_name already exists, they are combined under new_name
		if self._sections[new_name]:
			self._sections[new_name]["hits"] += self._sections[name]["hits"]
		else:
			self._sections[new_name] = self._sections[name]
		del self._sections[name]

	@staticmethod
	def thousand_seps(numstr: str or float or int) -> str:
		decs = str(numstr)
		rest = ""
		if "." in decs:
			rest = decs[decs.index("."):]
			decs = decs[:decs.index(".")]
		for i in range(len(decs)-3, 0, -3):
			decs = decs[:i] + "," + decs[i:]
		return decs + rest

	@classmethod
	def stringify_time(cls, dt: float, unit="ms"):
		str_ = f"{dt*cls._units[unit]:.3f} {unit}"
		return cls.thousand_seps(str_)
	
	def reset(self):
		self._sections = {}
		self._section_depth = 0

	def get_sections(self):
		# Returns data parts of sections
		return {kw: v for kw, v in self._sections.items() if kw != "tt"}

	def get_section_times(self):
		return {kw: np.sum(v["hits"]) for kw, v in self._sections.items()}

	def stringify_sections(self, unit="s"):
		# Returns pretty sections
		strs = [["Execution times", "Total time", "Hits", "Avg. time"]]
		std_strs = []
		for kw, v in self._sections.items():
			elapsed = np.sum(v["hits"])
			avg = elapsed / len(v["hits"])
			# std = self.stringify_time(2*np.std(v["hits"]), "ms")
			# std_strs.append(std)
			strs.append([
				"- " * v["depth"] + kw,
				self.stringify_time(elapsed, unit),
				self.thousand_seps(len(v["hits"])),
				self.stringify_time(avg, "ms")# + " p/m ",
			])
		# longest_std = max(len(x) for x in std_strs)
		# std_strs = [" " * (longest_std-len(x)) + x for x in std_strs]
		# for i, str_ in enumerate(strs[1:]): str_[-1] += std_strs[i]
		for i in range(len(strs[0])):
			length = max(len(strs[j][i]) for j in range(len(strs)))
			for j in range(len(strs)):
				if i == 0:
					strs[j][i] += " " * (length - len(strs[j][i]))
				else:
					strs[j][i] = " " * (length - len(strs[j][i])) + strs[j][i]
		for i in range(len(strs)):
			strs[i] = " | ".join(strs[i])
		return "\n".join(strs)

	def __str__(self):
		return self.stringify_sections("s")

if __name__ == "__main__":
	tt = TickTock()
	for i in range(100_000):
		tt.section("Test")
		tt.end_section("Test")
	print(tt)


