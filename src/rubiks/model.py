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

		shared_thiccness = [288, 4096, 2048]
		policy_thiccness = [shared_thiccness[-1], 512, 12]
		value_thiccness = [shared_thiccness[-1], 512, 1]
		self.shared_net = nn.Sequential(*self._create_fc_layers(shared_thiccness))
		self.policy_net = nn.Sequential(*self._create_fc_layers(policy_thiccness))
		self.value_net = nn.Sequential(*self._create_fc_layers(value_thiccness))

		self.log(f"Created network\n{self.config}\n{self}")

	def _create_fc_layers(self, thiccness: list):
		layers = []
		for i in range(len(thiccness)-1):
			layers.append(nn.Linear(thiccness[i], thiccness[i+1]))
			layers.append(nn.Dropout(self.config.dropout))
			if self.config.batchnorm:
				layers.append(nn.BatchNorm1d(thiccness[i+1]))

		return layers
	
	def forward(self, x, policy = True, value = True):
		assert policy or value
		x = self.shared_net(x)
		return_values = []
		if policy:
			policy = self.policy_net(x)
			return_values.append(policy)
		if value:
			value = self.value_net(x)
			return_values.append(value)
		return return_values if len(return_values) > 1 else return_values[0]
	
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
