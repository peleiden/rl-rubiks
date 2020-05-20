import os
import json
import torch

from tests import MainTest

from librubiks import cpu, gpu, set_is2024
from librubiks.model import Model, ModelConfig
from librubiks.utils import NullLogger


class TestModel(MainTest):
	def test_model(self):
		config = ModelConfig()
		model = Model.create(config).to(gpu)
		assert next(model.parameters()).device.type == gpu.type
		model.eval()
		x = torch.randn(2, 480).to(gpu)
		model(x)
		model.train()
		model(x)

	def test_resnet(self):
		config = ModelConfig(architecture = 'res')
		model = Model.create(config).to(gpu)
		assert next(model.parameters()).device.type == gpu.type
		model.eval()
		x = torch.randn(2, 480).to(gpu)
		model(x)
		model.train()
		model(x)

	def test_save_and_load(self):
		torch.manual_seed(42)

		config = ModelConfig()
		model = Model.create(config, logger=NullLogger()).to(gpu)
		model_dir = "local_tests/local_model_test"
		model.save(model_dir)
		assert os.path.exists(f"{model_dir}/config.json")
		assert os.path.exists(f"{model_dir}/model.pt")

		model = Model.load(model_dir).to(gpu)
		assert next(model.parameters()).device.type == gpu.type


	def test_model_config(self):
		cf = ModelConfig(torch.nn.ReLU())
		with open("local_tests/test_config.json", "w", encoding="utf-8") as f:
			json.dump(cf.as_json_dict(), f)
		with open("local_tests/test_config.json", encoding="utf-8") as f:
			cf = ModelConfig.from_json_dict(json.load(f))
		assert type(cf.activation_function) == type(torch.nn.ReLU())

	def test_init(self):
		for init in ['glorot', 'he', 0, 1.123123123e-3]:
			cf = ModelConfig(init=init)
			model = Model.create(cf).to(gpu)
			x = torch.randn(2,480).to(gpu)
			model(x)
