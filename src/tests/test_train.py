import os
import torch

from src.rubiks.train import Train
from src.rubiks.model import Model, ModelConfig


class TestTrain:

	def test_train(self):
		torch.manual_seed(42)
		net = Model(ModelConfig())
		optim = torch.optim.Adam(net.parameters())
		loss = torch.nn.CrossEntropyLoss()
		train = Train(optim, loss)
		net, val_epochs, train_losses, val_losses = train.train(net, 1000, 10, 10)
		
		train.plot_training(val_epochs, train_losses, val_losses, "local_train_test")
		assert os.path.exists("local_train_test/training.png")
		



