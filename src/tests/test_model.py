import os
import torch

from src.rubiks.model import Model, ModelConfig
from src.rubiks.utils.logger import NullLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestModel:
	
	def test_model(self):
		config = ModelConfig()
		model = Model(config)
		assert next(model.parameters()).device.type == device.type
	
	def test_save_and_load(self):
		torch.manual_seed(42)
		
		config = ModelConfig()
		model = Model(config, logger=NullLogger())
		model_dir = "local_model_test"
		model.save(model_dir)
		assert os.path.exists(f"{model_dir}/config.json")
		assert os.path.exists(f"{model_dir}/model.pt")
		
		model = Model.load(model_dir)
		assert next(model.parameters()).device.type == device.type
		

