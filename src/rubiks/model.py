import json
import os
import torch
import torch.nn as nn
from copy import deepcopy

from src.rubiks.cube.cube import Cube
from src.rubiks import cpu, gpu, get_is2024
from src.rubiks.utils.logger import Logger, NullLogger

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ModelConfig:
	activation_function: torch.nn.functional = torch.nn.ELU()
	batchnorm: bool = True
	architecture: str = 'fc'  # Options: 'fc', 'res', 'convo'

	# Hidden layer sizes in  shared network and in the two part networks given as tuples. If None: The value is controlled by architecture (often wanted)
	shared_sizes: tuple = None
	part_sizes: tuple = None
	conv_channels: tuple = None
	intermediate_sizes: tuple = None

	_fc_arch: ClassVar[dict] = {"shared_sizes": (4096, 2048), "part_sizes": (512,)}
	_res_arch: ClassVar[dict] = {"shared_sizes": (5000, 1000), "part_sizes": (100,)}
	_conv_arch: ClassVar[dict] = {"shared_sizes": (4096, 2048), "part_sizes": (512,), "conv_channels": (4, 4), "intermediate_sizes": (1024,)}

	def __post_init__(self):
		if self.shared_sizes is None:
			self.shared_sizes = self._get_arch()["shared_sizes"]
		if self.part_sizes is None:
			self.part_sizes = self._get_arch()["part_sizes"]
		if self.conv_channels is None and self.architecture == "conv":
			self.conv_channels = self._get_arch()["conv_channels"]
		if self.intermediate_sizes is None and self.activation_function == "conv":
			self.intermediate_sizes = self._get_arch()["intermediate_sizes"]

	def _get_arch(self):
		return getattr(self, f"_{self.architecture}_arch")

	@classmethod
	def _get_non_serializable(cls):
		return {"activation_function": cls._get_activation_function}

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
	def _get_activation_function(val, from_key: bool):
		afs = {"elu": torch.nn.ELU(), "relu": torch.nn.ReLU()}
		if from_key:
			return afs[val]
		else:
			return [x for x in afs if type(afs[x]) == type(val)][0]


class Model(nn.Module):
	"""
	A fully connected, feed forward Neural Network.
	Also the instantiator class of other network architectures through `create`.
	"""
	shared_net: nn.Sequential
	policy_net: nn.Sequential
	value_net: nn.Sequential

	def __init__(self, config: ModelConfig, logger=NullLogger()):
		super().__init__()
		self.config = config
		self.log = logger

		self._construct_net()
		self.log(f"Created network\n{self.config}\n{self}")

	@staticmethod
	def create(config: ModelConfig, logger=NullLogger()):
		"""
		Allows this class to be used to instantiate other Network architectures based on the content
		of the configuartion file.
		"""
		if config.architecture == "fc": return Model(config, logger)
		if config.architecture == "res": return ResNet(config, logger)
		if config.architecture == "conv": return ConvNet(config, logger)

		raise KeyError(f"Network architecture should be 'fc', 'res', or 'conv', but '{config.architecture}' was given")

	def _construct_net(self):
		"""
		Constructs a feed forward fully connected DNN.
		"""
		shared_thiccness = [Cube.get_oh_shape(), *self.config.shared_sizes]
		policy_thiccness = [shared_thiccness[-1], *self.config.part_sizes, Cube.action_dim]
		value_thiccness = [shared_thiccness[-1], *self.config.part_sizes, 1]

		self.shared_net = nn.Sequential(*self._create_fc_layers(shared_thiccness, False))
		self.policy_net = nn.Sequential(*self._create_fc_layers(policy_thiccness, True))
		self.value_net = nn.Sequential(*self._create_fc_layers(value_thiccness, True))

	def forward(self, x, policy=True, value=True):
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


	def _create_fc_layers(self, thiccness: list, final: bool):
		"""
		Helper function to return fully connected feed forward layers given a list of layer sizes and
		a final output size.
		"""
		layers = []
		for i in range(len(thiccness)-1):
			layers.append(nn.Linear(thiccness[i], thiccness[i+1]))
			if not (final and i == len(thiccness) - 2):
				layers.append(self.config.activation_function)
				if self.config.batchnorm:
					layers.append(nn.BatchNorm1d(thiccness[i+1]))

		return layers

	def clone(self):
		new_state_dict = {}
		for kw, v in self.state_dict().items():
			new_state_dict[kw] = v.cpu().clone()
		new_net = Model.create(self.config)
		new_net.load_state_dict(new_state_dict)
		return new_net

	def get_params(self):
		return torch.cat([x.float().flatten() for x in self.state_dict().values()]).clone()

	def save(self, save_dir: str, is_min=False):
		"""
		Save the model and configuration to the given directory
		The folder will include a pytorch model, and a json configuration file
		"""

		os.makedirs(save_dir, exist_ok=True)
		if is_min:
			model_path = os.path.join(save_dir, "model-min.pt")
			torch.save(self.state_dict(), model_path)
			self.log(f"Saved min model to {model_path}")
			return
		model_path = os.path.join(save_dir, "model.pt")
		torch.save(self.state_dict(), model_path)
		conf_path = os.path.join(save_dir, "config.json")
		with open(conf_path, "w", encoding="utf-8") as conf:
			json.dump(self.config.as_json_dict(), conf)
		self.log(f"Saved model to {model_path} and configuration to {conf_path}")

	@staticmethod
	def load(load_dir: str, logger=NullLogger()):
		"""
		Load a model from a configuration directory
		"""

		model_path = os.path.join(load_dir, "model.pt")
		conf_path = os.path.join(load_dir, "config.json")
		with open(conf_path, encoding="utf-8") as conf:
			state_dict = torch.load(model_path, map_location=gpu)
			config = ModelConfig.from_json_dict(json.load(conf))

		model = Model.create(config, logger)
		model.load_state_dict(state_dict)
		model.to(gpu)
		# First time the net is loaded, a feedforward is performed, as the first time is slow
		# This avoids skewing evaluation results
		with torch.no_grad():
			model.eval()
			model(Cube.as_oh(Cube.get_solved()))
			model.train()
		return model




class ResNet(Model):
	"""
	A Residual Neural Network.
	"""
	def _construct_net(self):
		raise NotImplementedError

class ConvNet(Model):

	shared_conv_net: nn.Sequential

	def _construct_net(self):
		super()._construct_net()
		conv_layers = []
		conv_layers = [nn.Conv1d(1, self.config.conv_channels[0], 1)]
		for channels in self.config.conv_channels[1:]:
			conv_layers.append(self.config.activation_function)
			# conv_layers.append(nn.BatchNorm1d())  # TODO: Calculate number of features
			conv_layers.append(nn.Conv1d())

	def forward(self, x, policy=True, value=True):
		assert policy or value
		padded_x = Cube.pad(x, len(self.config.conv_channels))
		raise NotImplementedError


