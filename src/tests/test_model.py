import os
import json
import torch

from src.rubiks.model import Model, ModelConfig
from src.rubiks.utils.logger import NullLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model_config():
	cf = ModelConfig(torch.nn.ReLU())
	with open("test_config.json", "w") as f:
		json.dump(cf.as_json_dict(), f)
	with open("test_config.json") as f:
		cf = ModelConfig.from_json_dict(json.load(f))
	os.remove("test_config.json")
	assert type(cf.activation_function) == type(torch.nn.ReLU())

class TestModel:
	
	def test_model(self):
		config = ModelConfig()
		model = Model(config)
		assert next(model.parameters()).device.type == device.type
		model.eval()
		x = torch.randn(2, 288)
		model(x)
		model.train()
		model(x)
	
	def test_save_and_load(self):
		torch.manual_seed(42)
		
		config = ModelConfig()
		model = Model(config, logger=NullLogger())
		model_dir = "local_tests/local_model_test"
		model.save(model_dir)
		assert os.path.exists(f"{model_dir}/config.json")
		assert os.path.exists(f"{model_dir}/model.pt")
		
		model = Model.load(model_dir)
		assert next(model.parameters()).device.type == device.type
		

