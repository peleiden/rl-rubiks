import matplotlib.pyplot as plt
import numpy as np
import os
from src.rubiks.utils.logger import Logger, NullLogger


class Train:
	def __init__(self, optim, loss_fn, logger = NullLogger()):
		self.optim = optim
		self.loss_fn = loss_fn
		self.log = logger
		self.log("Created trainer with", f"Optimizer: {self.optim}", f"Loss function: {self.loss_fn}\n")
		
	def train(self, net, epochs: int, validation_interval: int, batch_size: int):
		self.log(f"Beginning training", f"Epochs: {epochs}", f"Validation every: {validation_interval}",
				 f"Batch size: {batch_size}")
		n_vals = epochs // validation_interval
		val_epochs = np.empty(n_vals)
		train_losses = np.empty(n_vals)
		val_losses = np.empty(n_vals)
		for epoch in epochs:
			# TODO Perform training
			if (epoch + 1) % validation_interval == 0:
				# TODO Perform validation
				pass
		
		return net, val_epochs, train_losses, val_losses
	
	@staticmethod
	def plot_training(val_epochs, train_losses, val_losses, save_dir: str, title="", show=False):
		
		plt.figure(figsize=(19.2, 10.8))
		plt.plot(val_epochs, train_losses, "b", label="Training loss")
		plt.plot(val_epochs, val_losses, "r-", label="Validation loss")
		plt.title(title if title else "Training loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		os.path.makedirs(save_dir, exist_ok=True)
		plt.savefig(os.path.join(save_dir, "training.png"))
		if show:
			plt.show()
