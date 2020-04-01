import os, sys
os.chdir(sys.path[0])
from pathlib import Path
import git
import matplotlib.pyplot as plt
import numpy as np
repopath = os.path.realpath(os.path.join(os.getcwd(), "..", ".."))
os.chdir(repopath)
print(repopath)
repo = git.Repo(repopath)
commits = list(reversed([str(x) for x in repo.iter_commits()]))
n_commits = np.arange(1, len(commits)+1)
n_lines = np.zeros(len(commits))
for i, commit in enumerate(commits):
	cmd = f"git checkout {commit}"
	print(f">>> {cmd}")
	os.system(cmd)
	pyfiles = list(Path(".").rglob("*.[pP][yY]"))
	for pyfile in pyfiles:
		with open(str(pyfile)) as py:
			lines = [x.strip() for x in py.readlines()]
			for line in lines:
				if line and not line.startswith("#"):
					n_lines[i] += 1
print(n_lines)
os.system("git checkout master")



