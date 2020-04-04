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
patterns = {
	".py": ("*.[pP][yY]", "#"),
	".tex": ("*.[tT][eE][xX]", "%"),
	".md": ("*.[mM][dD]", "None"*10),
}
n_commits = np.arange(0, len(commits)+1)
n_lines = {kw: np.zeros(len(commits)+1) for kw in patterns}
for i, commit in enumerate(commits):
	cmd = f"git checkout {commit}"
	print(f"{i+1} / {len(commits)} >>> {cmd}")
	os.system(cmd)
	for kw, (pattern, comment) in patterns.items():
		files = list(Path(".").rglob(pattern))
		for p in files:
			with open(str(p)) as f:
				lines = [x.strip() for x in f.readlines()]
				for line in lines:
					if line and not line.startswith(comment):
						n_lines[kw][i+1] += 1

os.system("git checkout master")

plt.figure(figsize=(15, 10))
for kw, lines in n_lines.items():
	plt.plot(n_commits, lines, "-o", label=kw)
plt.xlabel("Number of commits")
plt.ylabel("Nummer of non-empty/comment lines")
plt.legend(loc=2)
plt.grid(True)
plt.savefig("line_counts.png")
plt.show()



