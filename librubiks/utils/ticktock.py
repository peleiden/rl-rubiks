from time import perf_counter
from typing import List, Dict, Tuple

class TimeUnit:
	nanosecond  = ("ns",  1e9)
	microsecond = ("mus", 1e6)
	millisecond = ("ms",  1e3)
	second      = ("s",   1)
	minute      = ("min", 1/60)
	hour        = ("h",   1/3600)

class Profile:

	start: float

	def __init__(self, name: str, depth: int):
		self.hits: List[float] = []
		self.name = name
		self.depth = depth
	
	def get_hits(self):
		return self.hits

	def sum(self):
		# Returns total runtime
		return sum(self.get_hits())

	def mean(self):
		# Returns mean runtime lengths
		return self.sum() / len(self) if self.get_hits() else 0
	
	def std(self):
		# Returns empirical standard deviation of runtime
		# Be aware that this is highly sensitive to outliers and often a bad estimate
		s = self.mean()
		return (1 / (len(self)+1) * sum(map(lambda x: (x-s)**2, self.get_hits()))) ** 0.5
	
	def remove_outliers(self, threshold=2):
		# Remove all hits larger than threshold * average
		# Returns number of removed outliers
		mu = self.mean()
		original_length = len(self)
		self.hits = [x for x in self.hits if x <= threshold * mu]
		return original_length - len(self)
	
	def __str__(self):
		return self.name

	def __len__(self):
		return len(self.hits)


class TickTock:

	_start = 0
	profiles: Dict[str, Profile] = {}
	_profile_depth = 0
	_latest_profile: str

	def tick(self):
		self._start = perf_counter()
		return self._start

	def tock(self):
		end = perf_counter()
		return end - self._start

	def profile(self, name: str):
		if name not in self.profiles:
			self.profiles[name] = Profile(name, self._profile_depth)
		self._profile_depth += 1
		self._latest_profile = name
		self.profiles[name].start = perf_counter()

	def end_profile(self, name: str=None):
		end = perf_counter()
		name = name or self._latest_profile
		dt = end - self.profiles[name].start
		self.profiles[name].hits.append(dt)
		self._profile_depth -= 1
		return dt

	def rename_section(self, name: str, new_name: str):
		# Renames a section
		# If a section with new_name already exists, they are combined under new_name
		# TODO: Drop this method and make a fuse function instead
		if self.profiles[new_name]:
			self.profiles[new_name].hits += self.profiles[name].hits
		else:
			self.profiles[new_name] = self.profiles[name]
		del self.profiles[name]

	def remove_outliers(self, threshold=2):
		# For all profiles, remove hits longer than threshold * average hit
		for profile in self.profiles.values():
			profile.remove_outliers(threshold)

	def reset(self):
		self.profiles = {}
		self._profile_depth = 0

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
	def stringify_time(cls, dt: float, unit: Tuple[str, float]=TimeUnit.millisecond):
		str_ = f"{dt*unit[1]:.3f} {unit[0]}"
		return cls.thousand_seps(str_)

	def stringify_sections(self, unit: Tuple[str, float]=TimeUnit.second):
		# Returns pretty sections
		strs = [["Execution times", "Total time", "Hits", "Avg. time"]]
		# std_strs = []
		for kw, v in self.profiles.items():
			# std = self.stringify_time(2*np.std(v["hits"]), "ms")
			# std_strs.append(std)
			strs.append([
				"- " * v.depth + kw,
				self.stringify_time(v.sum(), unit),
				self.thousand_seps(len(v)),
				self.stringify_time(v.mean(), TimeUnit.millisecond)# + " p/m ",
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
		return self.stringify_sections(TimeUnit.second)

