import os
import torch

from src.rubiks.train import Train
from src.rubiks.model import Model, ModelConfig
from src.rubiks.utils import cpu, gpu
from src.rubiks.solving.agents import PolicyCube

class TestTrain:

	def test_train(self):
		torch.manual_seed(42)

		#The standard test
		net = Model(ModelConfig()).to(gpu)
		# TODO: Update to refactored train class
		train = Train(rollouts=1, batch_size=2, rollout_games=2, rollout_depth=3, optim_fn=torch.optim.Adam, agent=PolicyCube, lr=1e-6,
				)

		# Current
		net = train.train(net)

		train.plot_training("local_tests/local_train_test")
		assert os.path.exists("local_tests/local_train_test/training.png")

		# optim = torch.optim.Adam
		# policy_loss = torch.nn.CrossEntropyLoss
		# val_loss = torch.nn.MSE
