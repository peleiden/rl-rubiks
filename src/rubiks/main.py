
from src.pipeline.pipeline import Pipeline
from src.pipeline.jobs import jobs

if __name__ == "__main__":
	# TODO: Add argparser and create pipeline from it
	# If no args are given, jobs.py should be used (alternatively, with --jobs flag)
	Pipeline.exec(jobs)
