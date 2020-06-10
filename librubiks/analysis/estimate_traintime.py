import os
import shutil
from datetime import timedelta

import numpy as np

from runtrain import options
from librubiks.jobs import TrainJob
from librubiks.utils import set_seeds, Logger, Parser, TickTock

if __name__ == "__main__":
	set_seeds()
	parser = Parser(options, description="Estimate the amount of times required for given jobs", name="train")
	estimated_runtime = 0
	tt = TickTock()
	job_settings = parser.parse(False)
	for settings in job_settings:
		job_rollouts = settings["rollouts"]
		job_evaluation_interval = settings["evaluation_interval"]
		settings["rollouts"] = 5  # Five rollouts should be good enough to give a decent estimate
		settings["evaluation_interval"] = 0
		# Estimates training time
		tt.tick()
		train = TrainJob(**settings)
		train.execute()
		estimated_runtime += tt.tock() * job_rollouts / settings["rollouts"]
		# Estimates evaluation time
		evaluations = job_rollouts / job_evaluation_interval if job_evaluation_interval else 0
		estimated_runtime += np.ceil(evaluations) * TrainJob.eval_games * TrainJob.max_time

		# Cleans up
		shutil.rmtree(settings["location"])

	log_loc = job_settings[0]["location"]\
		if len(job_settings) == 1\
		else os.path.abspath(os.path.join(job_settings[0]["location"], ".."))
	log_loc += "/runtime_estimation.txt"
	log = Logger(log_loc, "Training time estimation")
	log("\n".join([
		f"Expected training time for the {len(job_settings)} given jobs: {timedelta(seconds=int(estimated_runtime))}",
		f"With 20 % buffer: {timedelta(seconds=int(estimated_runtime*1.2))}"
	]))

