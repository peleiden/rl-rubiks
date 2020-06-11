import os, sys
path = sys.path[0]
import git
os.chdir(path)
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14})
import numpy as np
from collections import deque, defaultdict
import time

def exclude(d: str):
	exclude_patterns = ["local", "node_modules", "dist", ".idea", "__pycache__", ".git", ".pytest_cache", ".vscode", "legacy", "dev"]
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


def search_repo(repopath: str, branch: str, patterns: dict) -> (list, dict, list):
	os.chdir(repopath)
	print(repopath)
	repo = git.Repo(repopath)
	commits = list(reversed(list(repo.iter_commits())))
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
						if line and not (patterns[ext][1] and line.startswith(patterns[ext][1])):
							n_lines[ext][i+1] += 1

	os.system("git checkout " + branch)
	return times, n_lines, commits

def fuse(t1: list, t2: list, n_lines1: dict, n_lines2: dict, commits1: list, commits2: list) -> (list, dict, list):
	"""Fuses two repo countings"""
	assert n_lines1.keys() == n_lines2.keys(), "Extensions must match"
	times = []
	n_lines = { kw: np.zeros(len(commits1)+len(commits2)+1) for kw in n_lines1.keys() }
	commits = []
	i, j = 0, 0
	while i < len(t1) or j < len(t2):
		k = i + j + 1
		if i < len(t1) and (j == len(t2) or t1[i] < t2[j]):
			times.append(t1[i])
			commits.append(commits1[i])
			for ext in n_lines1.keys():
				n_lines[ext][k] = n_lines1[ext][i] + n_lines2[ext][j]
			i += 1
		else:
			times.append(t2[j])
			commits.append(commits2[j])
			for ext in n_lines2.keys():
				n_lines[ext][k] = n_lines1[ext][i] + n_lines2[ext][j]
			j += 1

	return times, n_lines, commits


def remove_outliers(exts: list, n_lines: dict):
	# There are some unfortunate outliers in .html (angular boilerpate) and .tex (twice the repo, double the lines)
	# The outliers are removed here by rather crudely assuming that only outliers see an increase of over 300 lines commit to commit
	for ext in exts:
		for i in range(1, len(n_lines[ext])):
			if n_lines[ext][i] > n_lines[ext][i-1] + 300:
				n_lines[ext][i] = n_lines[ext][i-1]
	return n_lines


def plot(times: list, n_lines: dict, commits: list, save_path: str):
	plt.figure(figsize=(15, 10))
	for kw, lines in n_lines.items():
		plt.plot(times, lines[1:], "-o", label=kw)
	xticks = np.linspace(0, len(commits)-1, 10, dtype=int)
	tickcommits = [x for i, x in enumerate(commits) if i in xticks]
	xticklabels = [time.strftime("%d-%m-%Y", time.gmtime(x.committed_date)) for x in tickcommits]
	plt.xticks([times[i] for i in xticks], xticklabels, rotation=60)
	plt.title("Development of line counts over time")
	plt.xlabel("Date of commit")
	plt.ylabel("Nummer of lines (excl. empty lines and comments)")
	plt.legend(loc=2)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(save_path, "local_line_counts.png"))
	plt.clf()


if __name__ == "__main__":
	patterns = {  # { label: ( extension, comment ) }
		".py":   (".py",   "#"),
		".ts":   (".ts",   "//"),
		".html": (".html", "<!--"),
		".tex":  (".tex",  "%"),
		".md":   (".md",   "")
	}
	repos = [(os.path.realpath(path + "/.."), "develop")]
	if os.path.exists("../../professorpatruljen"):
		repos.append((os.path.realpath("../../professorpatruljen"), "master"))
	times = []
	n_lines = []
	commits = []
	for repo, branch in repos:
		print(os.getcwd())
		res = search_repo(repo, branch, patterns)
		times.append(res[0])
		n_lines.append(res[1])
		commits.append(res[2])
		os.chdir(path)
	if len(repos) >= 2:
		times, n_lines, commits = fuse(times[0], times[1], n_lines[0], n_lines[1], commits[0], commits[1])
	else:
		times, n_lines, commits = times[0], n_lines[0], commits[0]
	n_lines = remove_outliers([".html", ".tex", ".md"], n_lines)
	plot(times, n_lines, commits, path)

