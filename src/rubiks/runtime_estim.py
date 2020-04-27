import shutil

from src.rubiks.runtrain import TrainJob, options
from src.rubiks.utils import seedsetter
from src.rubiks.utils.parse import Parser
from src.rubiks.utils.ticktock import TickTock

if __name__ == "__main__":
	seedsetter()
	parser = Parser(options, description="Estimate the amount of times required for given jobs", name="train")
	estimated_runtime = 0
	tt = TickTock()
	for settings in parser.parse():
		job_rollouts = settings["rollouts"]
		settings["rollouts"] = 5  # Five rollouts should be good enough to give a decent estimate
		# Estimates training time
		tt.tick()
		train = TrainJob(**settings)
		train.execute()
		estimated_runtime += tt.tock() * job_rollouts / settings["rollouts"]
		# Estimates evaluation time

		# Cleans up
		shutil.rmtree(settings["location"])
	log



