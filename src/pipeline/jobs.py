from dataclasses import dataclass
from pprint import pformat
import numpy as np
import torch

from src.rubiks.model import ModelConfig
from src.rubiks.solving.agents import DeepAgent, PolicyCube
from src.rubiks.utils.logger import Logger
from src.rubiks.train import Train
from src.rubiks.solving.evaluation import Evaluator

@dataclass
class Job:
	loc: str
	train_args: dict  # Should match arguments for Train.__init__ excluding logger
	model_cfg: ModelConfig
	title: str = None  # Defaults to loc
	eval_args: dict = None  # Should match arguments for Evaluator.__init__ excluding logger
	agents: tuple = ()  # Tuple of functions creating agents. Should only take net as arg. Evaluation will be performed for each agent
	is2024: bool = True
	verbose: bool = True
	def __post_init__(self):
		self.title = self.title or self.loc

	def __str__(self):
		return pformat(self.__dict__)

jobs = [
	Job(
		loc = "local_pipeline_test",
		title = "Pipeline test",
		train_args = {
			 "rollouts": 1,
			 "batch_size": 5,  # Required to be > 1 when training with batchnorm
			 "rollout_games": 10,
			 "rollout_depth": 20,
			 "optim_fn": torch.optim.RMSprop,
			 "lr": 1e-5,
			 "policy_criterion": torch.nn.CrossEntropyLoss,
			 "value_criterion": torch.nn.MSELoss,
		},
		model_cfg = ModelConfig(),
		agents = (lambda net: PolicyCube(net, True), ),
	),
]

