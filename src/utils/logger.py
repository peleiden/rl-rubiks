import os

from ticktock import get_timestamp

class Logger:

	def __init__(self, fpath: str, title: str):
		dirs = "".join(fpath.split('/')[:-1])
		if not os.path.exists(dirs):
			os.makedirs(dirs)

		self.fpath = fpath
	
		with open(self.fpath, "w+", encoding="utf-8") as logfile:
			logfile.write("")
		
		self.log(title + "\n")
	
	def __call__(self, *tolog, with_timestamp=True):

		self.log(*tolog, with_timestamp=with_timestamp)
	
	def log(self, *tolog, with_timestamp=True):

		time = get_timestamp()
		with open(self.fpath, "a") as logfile:
			tolog = " ".join([str(x) for x in tolog])
			n_spaces = len(time)
			logs = tolog.split("\n")
			if with_timestamp:
				logs[0] = time + "\t" + logs[0]
			else:
				logs[0] = n_spaces * " " + "\t" + logs[0]
			for i in range(1, len(logs)):
				logs[i] = n_spaces * " " + "\t" + logs[i]
			if logs[i].strip() == "":
				logs[i] = ""
			tolog = "\n".join(logs)
			logfile.write(tolog+"\n")
			print(tolog)
	
	def newline(self):
		with open(self.fpath, "a", encoding="utf-8") as logfile:
			logfile.write("\n")
			print()

class NullLogger(Logger):

	def __init__(self, *args, **kwargs):

		pass

	def log(self, *tolog, **kwargs):

		pass

	def newline(self):

		pass

