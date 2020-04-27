import os
import shutil
from datetime import timedelta

from src.rubiks.runtrain import TrainJob, options
from src.rubiks.utils import seedsetter
from src.rubiks.utils.logger import Logger
from src.rubiks.utils.parse import Parser
from src.rubiks.utils.ticktock import TickTock

if __name__ == "__main__":
	seedsetter()
	parser = Parser(options, description="Estimate the amount of times required for given jobs", name="train")
	estimated_runtime = 0
	tt = TickTock()
	job_settings = parser.parse(False)
	for settings in job_settings:
		print(settings["location"])
		job_rollouts = settings["rollouts"]
		settings["rollouts"] = 5  # Five rollouts should be good enough to give a decent estimate
		# Estimates training time
		tt.tick()
		train = TrainJob(**settings)
		train.execute()
		estimated_runtime += tt.tock() * job_rollouts / settings["rollouts"]
		# Estimates evaluation time
		estimated_runtime += settings["evaluations"] * TrainJob.eval_games * TrainJob.max_time

		# Cleans up
		shutil.rmtree(settings["location"])

	log_loc = job_settings[0]["location"]\
		if len(job_settings) == 1\
		else os.path.abspath(os.path.join(job_settings[0]["location"], ".."))
	log_loc += "/runtime_estimation.txt"
	log = Logger(log_loc, "Training time estimation")
	log("\n".join([
		f"Expected runtime for the {len(job_settings)} jobs given: {timedelta(seconds=int(estimated_runtime))}",
		f"With 20 % buffer: {timedelta(seconds=int(estimated_runtime*1.2))}"
	]))

