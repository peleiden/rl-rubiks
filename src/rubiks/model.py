import json
import os
import zipfile
import torch
import torch.nn as nn
from copy import deepcopy

from src.rubiks.utils.logger import Logger, NullLogger

from dataclasses import dataclass


@dataclass
class ModelConfig:
	activation_function: torch.nn.functional = torch.nn.ELU()
	dropout: float = 0
	batchnorm: bool = True

	@classmethod
	def _get_non_serializable(cls):
		return {"activation_function": cls._conv_activation_function}

	def as_json_dict(self):
		d = deepcopy(self.__dict__)
		for a, f in self._get_non_serializable().items():
			d[a] = f(d[a], False)
		return d
	
	@classmethod
	def from_json_dict(cls, conf: dict):
		for a, f in cls._get_non_serializable().items():
			conf[a] = f(conf[a], True)
		return ModelConfig(**conf)

	@staticmethod
	def _conv_activation_function(val, from_key: bool):
		afs = {"elu": torch.nn.ELU(), "relu": torch.nn.ReLU()}
		if from_key:
			return afs[val]
		else:
			return [x for x in afs if type(afs[x]) == type(val)][0]





class Model(nn.Module):
	
	def __init__(self, config: ModelConfig, logger=NullLogger(),):
		super().__init__()
		self.config = config
		self.log = logger
		
		# Temporary model
		self.net = nn.Linear(6*8*6, 7)

		self.log(self)
	
	def forward(self, x):
		return self.net(x)
	
	def save(self, save_dir: str):
		"""
		Save the model and configuration to the given directory
		The folder will include a pytorch model, and a json configuration file
		A zip is also included
		"""
		
		os.makedirs(save_dir, exist_ok=True)
		model_path = os.path.join(save_dir, "model.pt")
		conf_path = os.path.join(save_dir, "config.json")
		with open(conf_path, "w", encoding="utf-8") as conf:
			torch.save(self.state_dict(), model_path)
			json.dump(self.config.as_json_dict(), conf)
	
	@staticmethod
	def load(load_dir: str):
		"""
		Load a model from a configuration directory
		"""
		
		model_path = os.path.join(load_dir, "model.pt")
		conf_path = os.path.join(load_dir, "config.json")
		with open(conf_path, encoding="utf-8") as conf:
			state_dict = torch.load(model_path)
			config = ModelConfig(**json.load(conf))
		
		model = Model(config)
		model.load_state_dict(state_dict)
		return model
