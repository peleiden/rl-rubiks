import os
import torch

from src.rubiks.train import Train
from src.rubiks.model import Model, ModelConfig
from src.rubiks.utils.device import cpu, gpu


class TestTrain:

	def test_train(self):
		torch.manual_seed(42)

		#The standard test		
		net = Model(ModelConfig()).to(gpu)
		train = Train()

		# Current
		net = train.train(net, 2, batch_size=2, rollout_games=2, rollout_depth=2, evaluation_interval=0)
		
		train.plot_training("local_tests/local_train_test")
		assert os.path.exists("local_tests/local_train_test/training.png")

		# optim = torch.optim.Adam
		# policy_loss = torch.nn.CrossEntropyLoss
		# val_loss = torch.nn.MSE