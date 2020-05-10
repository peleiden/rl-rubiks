import os, sys
os.chdir(sys.path[0])
from pathlib import Path
import git
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14})
import numpy as np
from collections import deque, defaultdict
import time

def exclude(d: str):
	exclude_patterns = ["local", "node_modules", "dist", ".idea", "__pycache__", ".git", ".pytest_cache", ".vscode"]
	for pattern in exclude_patterns:
		if pattern in d or f"/{pattern}" in d:
			return True
	return False

def get_files(patterns: dict):
	files = defaultdict(list)
	dirs = lambda x: next(os.walk(x))[1]
	q = deque(".")
	for f in os.listdir(q[0]):
		for ext in (x[0] for x in patterns.values()):
			if f.lower().endswith(ext):
				files[ext].append(os.path.join(q[0], f))
	while q:
		v = q.popleft()
		for d in dirs(v):
			d = os.path.join(v, d)
			if exclude(d): continue
			q.append(d)
			for f in os.listdir(d):
				for ext in (x[0] for x in patterns.values()):
					if f.lower().endswith(ext):
						files[ext].append(os.path.join(d, f))
	return files


repopath = os.path.realpath(os.path.join(os.getcwd(), "..", "..", ".."))
os.chdir(repopath)
print(repopath)
repo = git.Repo(repopath)
commits = list(reversed(list(repo.iter_commits())))
patterns = {
	".py": (".py", "#"),
	".ts": (".ts", "//"),
	".tex": (".tex", "%"),
	".md": (".md", "None"*10),
}
times = []
n_lines = {kw: np.zeros(len(commits)+1) for kw in patterns}

for i, commit in enumerate(commits):
	times.append(commit.committed_date)
	cmd = f"git checkout {commit}"
	print(f"{i+1} / {len(commits)} >>> {cmd}")
	os.system(cmd)
	files = get_files(patterns)
	for ext, fs in files.items():
		for p in fs:
			with open(str(p), encoding="utf-8") as f:
				lines = [x.strip() for x in f.readlines()]
				for line in lines:
					if line and not line.startswith(patterns[ext][1]):
						n_lines[ext][i+1] += 1

os.system("git checkout master")

plt.figure(figsize=(15, 10))
for kw, lines in n_lines.items():
	plt.plot(times, lines[1:], "-o", label=kw)
xticks = np.linspace(0, len(commits)-1, 10, dtype=int)
tickcommits = [x for i, x in enumerate(commits) if i in xticks]
xticklabels = [time.strftime("%d-%m-%Y", time.gmtime(x.committed_date)) for x in tickcommits]
plt.xticks([times[i] for i in xticks], xticklabels, rotation=60)
plt.xlabel("Date of commit")
plt.ylabel("Nummer of lines (excl. empty lines and comments)")
plt.legend(loc=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{repopath}/dev/legacy/project_management/local_line_counts.png")
plt.show()



