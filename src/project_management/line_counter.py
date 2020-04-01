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
n_commits = np.arange(0, len(commits)+1)
n_pylines = np.zeros(len(commits)+1)
n_texlines = np.zeros(len(commits)+1)
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
					n_pylines[i+1] += 1
	texfiles = list(Path(".").rglob("*.[tT][eE][xX]"))
	for texfile in texfiles:
		with open(str(texfile)) as tex:
			lines = [x.strip() for x in tex.readlines()]
			for line in lines:
				if line and not line.startswith("%"):
					n_texlines[i+1] += 1

os.system("git checkout master")

plt.figure(figsize=(15, 10))
plt.plot(n_commits, n_pylines, "-o", label=".py")
plt.plot(n_commits, n_texlines, "-o", label=".tex")
plt.xlabel("Number of commits")
plt.ylabel("Nummer of non-empty/comment lines")
plt.legend(loc=2)
plt.grid(True)
plt.show()



