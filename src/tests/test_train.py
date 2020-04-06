import os
import torch

from src.rubiks.train import Train
from src.rubiks.model import Model, ModelConfig
from src.rubiks import cpu, gpu
from src.rubiks.solving.agents import Agent, DeepAgent
from src.rubiks.solving.search import PolicySearch
from src.rubiks.solving.evaluation import Evaluator
class TestTrain:

	def test_train(self):
		torch.manual_seed(42)

		#The standard test
		net = Model(ModelConfig()).to(gpu)
		evaluator = Evaluator(2, 2, [2])
		train = Train(rollouts=2, batch_size=2, rollout_games=2, rollout_depth=3, optim_fn=torch.optim.Adam, agent=DeepAgent(PolicySearch(None)), lr=1e-6, evaluations=1, evaluator=evaluator)

		# Current
		net = train.train(net)

		train.plot_training("local_tests/local_train_test")
		assert os.path.exists("local_tests/local_train_test/training.png")

		# optim = torch.optim.Adam
		# policy_loss = torch.nn.CrossEntropyLoss
		# val_loss = torch.nn.MSE
