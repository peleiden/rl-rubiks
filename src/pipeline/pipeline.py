from src.pipeline.jobs import Job, jobs
from src.rubiks import get_repr, set_repr
from src.rubiks.utils.logger import Logger
from src.rubiks.utils import cpu, gpu
from src.rubiks.model import Model
from src.rubiks.train import Train
from src.rubiks.solving.evaluation import Evaluator

def exec_jobs():
	for job in jobs:
		exec_job(job)

def exec_job(job: Job):
	logger = Logger(f"{job.loc}/process.log", job.title, job.verbose)
	logger(f"Starting job:\n{job}")
	ini_repr = get_repr()
	set_repr(job.is2024)

	# Training
	logger.section()
	train = Train(**job.train_args, logger=logger)
	net = Model(job.model_cfg, logger).to(gpu)
	net = train.train(net)
	net.save(job.loc)
	train.plot_training(job.loc, f"Training of {job.title}")

	# Evaluation
	if job.eval_args:
		logger.section()
		evaluator = Evaluator(**job.eval_args, logger=logger)
		for agent_fn in job.agents:
			agent = agent_fn(net)

	# TODO: Finish implementing evaluation and return both training and evaluation results
	set_repr(ini_repr)
	return train.train_rollouts, train.train_losses

if __name__ == "__main__":
	exec_jobs()

