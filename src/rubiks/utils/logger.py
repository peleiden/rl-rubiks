import os

from src.rubiks.utils.ticktock import get_timestamp

class Logger:

	def __init__(self, fpath: str, title: str, verbose=True):
		dirs = "/".join(fpath.split('/')[:-1])
		if not os.path.exists(dirs):
			os.makedirs(dirs)

		self.fpath = fpath
		self._verbose = verbose
	
		with open(self.fpath, "w+", encoding="utf-8") as logfile:
			logfile.write("")
		
		self.log(title + "\n")
	
	def __call__(self, *tolog, with_timestamp=True):

		self.log(*tolog, with_timestamp=with_timestamp)
	
	def log(self, *tolog, with_timestamp=True):

		time = get_timestamp()
		with open(self.fpath, "a") as logfile:
			tolog = " ".join([str(x) for x in tolog])
			spaces = len(time) * " "
			logs = tolog.split("\n")
			if with_timestamp and tolog:
				logs[0] = f"{time}\t{logs[0]}"
			else:
				logs[0] = f"{spaces}\t{logs[0]}"
			for i in range(1, len(logs)):
				logs[i] = f"{spaces}\t{logs[i]}"
				if logs[i].strip() == "":
					logs[i] = ""
			tolog = "\n".join(logs)
			logfile.write(tolog+"\n")
			print(tolog)
	
	def verbose(self, *tolog, with_timestamp=True):
		if self._verbose:
			self(*tolog, with_timestamp=with_timestamp)
	
	def is_verbose(self):
		return self._verbose
	
	def section(self, title=""):
		self.log()
		self.log(title)

class NullLogger(Logger):
	
	_verbose = False

	def __init__(self, *args, **kwargs):
		pass

	def log(self, *tolog, **kwargs):
		pass

	def section(self, title=""):
		pass

