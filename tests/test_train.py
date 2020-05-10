import os
import torch

from tests import MainTest

from librubiks.train import Train
from librubiks.model import Model, ModelConfig
from librubiks import cpu, gpu
from librubiks.solving.agents import Agent, DeepAgent
from librubiks.solving.search import PolicySearch
from librubiks.solving.evaluation import Evaluator
class TestTrain(MainTest):

	def test_train(self):
		torch.manual_seed(42)
		#The standard test
		net = Model.create(ModelConfig()).to(gpu)
		evaluator = Evaluator(2, max_time=.02, max_states=None, scrambling_depths=[2])
		agent = DeepAgent(PolicySearch(None))
		train = Train(rollouts=2, batch_size=2, tau=0.1,  alpha_update =.5, gamma=1, rollout_games=2, rollout_depth=3, optim_fn=torch.optim.Adam, agent=agent, lr=1e-6, evaluation_interval=1, evaluator=evaluator, update_interval= 1, with_analysis=True)

		# Current
		net, min_net = train.train(net)

		train.plot_training("local_tests/local_train_test")
		assert os.path.exists("local_tests/local_train_test/training.png")

		# optim = torch.optim.Adam
		# policy_loss = torch.nn.CrossEntropyLoss
		# val_loss = torch.nn.MSE
