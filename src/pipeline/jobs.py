from dataclasses import dataclass
from pprint import pformat

from src.rubiks.utils.logger import Logger
from src.rubiks.train import Train
from src.rubiks.post_train.evaluation import Evaluator

@dataclass
class Job:
	loc: str
	train_args: dict  # Should match arguments for Train.__init__ excluding logger
	eval_args: dict  # Should match arguments for Evaluator.__init__ excluding logger
	
	def __str__(self):
		return pformat(self.__dict__)

jobs = [

]

