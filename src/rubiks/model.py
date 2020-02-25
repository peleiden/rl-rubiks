import json
import os
import zipfile
import torch
import torch.nn as nn

from src.rubiks.utils.logger import Logger, NullLogger

from dataclasses import dataclass

activation_functions = {
	"elu": torch.nn.ELU(),
	"relu": torch.nn.ReLU(),
	"softmax": torch.nn.Softmax(),
}
devices = {
	"cpu": torch.device("cpu"),
	"cuda": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


@dataclass
class ModelConfig:
	activation_function: str = "elu"
	device: str = "cuda"
	
	@staticmethod
	def from_dict(conf: dict):
		return ModelConfig(**conf)


class Model(nn.Module):
	
	def __init__(self, config: ModelConfig, logger=NullLogger(),):
		super().__init__()
		self.config = config
		self.log = logger
		
		# Temporary model
		self.net = nn.Linear(6*8*6, 7)
		
		self.to(devices[config.device])
	
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
			json.dump(self.config.__dict__, conf)
	
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
