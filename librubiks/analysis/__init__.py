import os

import numpy as np
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.animation as animations

from librubiks import gpu
from librubiks.cube import Cube
from librubiks.model import Model
from librubiks.utils import NullLogger, Logger

try:
	import networkx
	import imageio
	has_image_tools = True
except ModuleNotFoundError:
	has_image_tools = False

class TrainAnalysis:
	"""Performs analysis of the training procedure to understand loss and training behaviour"""
	def __init__(self, evaluations: np.ndarray, games: int, depth: int, extra_evals: int, logger: Logger=NullLogger()):
		"""Initialize containers mostly

		:param np.ndarray evaluations:  array of the evaluations performed on the model. Used for the more intensive analysis
		:param int depth: Rollout depth
		:param extra_evals: If != 0, extra evaluations are added for the first `exta_evals` rollouts

		"""

		self.games = games
		self.depth = depth
		self.depths = np.arange(1, depth)
		self.extra_evals = min(evaluations[-1] if len(evaluations) else 0, extra_evals) #Wont add evals in the future (or if no evals are needed)
		self.evaluations = np.unique( np.append(evaluations, range( self.extra_evals )) )

		self.orig_params = None
		self.params = None

		self.first_states = np.stack((
				Cube.get_solved(),
				*Cube.multi_rotate(Cube.repeat_state(Cube.get_solved(), Cube.action_dim), *Cube.iter_actions())
				))
		self.first_states = Cube.as_oh( self.first_states )
		self.first_state_values = list()

		self.substate_val_stds = list()

		self.avg_value_targets = list()
		self.param_changes = list()
		self.param_total_changes = list()

		self.policy_entropies = list()
		self.rollout_policy = list()

		self.log = logger
		self.log.verbose(f"Analysis of this training was enabled. Extra analysis is done for evaluations and for first {extra_evals} rollouts")

	def rollout(self, net: Model, rollout: int, value_targets: torch.Tensor):
		"""Saves statistics after a rollout has been performed for understanding the loss development

		:param torch.nn.Model net: The current net, used for saving values and policies of first 12 states
		:param rollout int: The rollout number. Used to determine whether it is evaluation time => check targets
		:param torch.Tensor value_targets: Used for visualizing value change
		"""
		# First time
		if self.params is None: self.params = net.get_params()

		# Keeping track of the entropy off on the 12-dimensional log-probability policy-output
		entropies = [entropy(policy, axis=1) for policy in self.rollout_policy]
		#Currently:  Mean over all games in entire rollout. Maybe we want it more fine grained later.
		self.policy_entropies.append(np.mean( [entropy.mean() for entropy in entropies] ))
		self.rollout_policy = list() #reset for next rollout

		if rollout in self.evaluations:
			net.eval()

			# Calculating value targets
			targets = value_targets.cpu().numpy()
			self.avg_value_targets.append(np.empty_like(self.depths, dtype=float))
			for i, depth in enumerate(self.depths):
				idcs = np.arange(self.games) * self.depth + depth
				self.avg_value_targets[-1][i] = targets[idcs].mean()

			# Calculating model change
			model_change = torch.sqrt((net.get_params()-self.params)**2).mean().cpu()
			model_total_change = torch.sqrt((net.get_params()-self.orig_params)**2).mean().cpu()
			self.params = net.get_params()
			self.param_changes.append(float(model_change))
			self.param_total_changes.append(model_total_change)

			#In the beginning: Calculate value given to first 12 substates
			if rollout <= self.extra_evals:
				self.first_state_values.append( net(self.first_states, policy=False, value=True).detach().cpu().numpy() )

			net.train()
	def ADI(self, values: torch.Tensor):
		"""Saves statistics after a run of ADI. """
		self.substate_val_stds.append(
			float(values.std(dim=1).mean())
		)


	def plot_substate_distributions(self, loc: str, show=False):
		self.log("Making plot of policy entropy and ADI value stds")

		fig, entropy_ax = plt.subplots(figsize=(19.2, 10.8))
		entropy_ax.set_xlabel(f"Rollout number")

		colour = "red"
		entropy_ax.set_ylabel(f"Rollout mean Shannon entropy", color=colour)
		entropy_ax.plot(self.policy_entropies, linestyle="dashdot", label="Entropy of training policy output for cubes", color=colour)
		entropy_ax.tick_params(axis='y', labelcolor = colour)
		h1, l1 = entropy_ax.get_legend_handles_labels()

		colour = 'blue'
		std_ax = entropy_ax.twinx()
		std_ax.set_ylabel(f"Rollout mean std.", color=colour)
		std_ax.plot(self.substate_val_stds, linestyle="dashdot", color=colour, label="Std. for ADI substates for cubes")
		std_ax.tick_params(axis='y', labelcolor=colour)

		h2, l2 = std_ax.get_legend_handles_labels()

		entropy_ax.legend(h1+h2, l1+l2)

		fig.tight_layout()
		plt.title(f"Analysis of substate distributions over time")
		plt.grid(True)

		path = os.path.join(loc, "substate_dists.png")
		plt.savefig(path)
		if show: plt.show()
		plt.clf()

		self.log(f"Saved substate probability plot to {path}")

	def visualize_first_states(self, loc: str):
		if has_image_tools and self.evaluations:
			self.log("Making visualization of first state values")
			gif_frames = []

			# Build graph structure
			G = networkx.DiGraph()
			edge_labels = {}
			G.add_nodes_from(range(len(self.first_state_values[0])))
			positions = {0: (50, 85)}
			label_positions = {0: (50, 80)}
			# Labels must be
			for i in range(Cube.action_dim):
				x_pos = 100*( i / (Cube.action_dim - 1) )
				positions[i+1] = (x_pos, 5)
				label_positions[i+1] = (x_pos, 12.5)

			for i, (face, pos) in enumerate(Cube.action_space):
				G.add_edge(0, i+1)
				edge_labels[(0, i+1)] =	Cube.action_names[face].lower() if pos else Cube.action_names[face].upper()



			fig = plt.figure(figsize=(10, 7.5))
			for i, values in enumerate(self.first_state_values):

				plt.title(f"Values at rollout:  {self.evaluations[i]}")

				labels = {j: f"{float(val):.2f}" for j, val in enumerate(values)}
				colors = [float(val) for val in values] #Don't ask
				networkx.draw(G, pos=positions, alpha=0.8, node_size=1000, \
						cmap = plt.get_cmap('cool'), node_color=colors, vmin=-1, vmax=1.5)

				networkx.draw_networkx_labels(G, pos=label_positions, labels=labels, font_size = 15)
				networkx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels,\
						font_size = 22, label_pos=0.25)

				plt.axis('off')
				fig.tight_layout()
				# https://stackoverflow.com/a/57988387, but is there an easier way?
				fig.canvas.draw()
				image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
				image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
				gif_frames.append(image_from_plot)

				plt.clf()

			if len(gif_frames) > 3: gif_frames.extend(gif_frames[-1] for i in range(10)) # Hacky way to pause gif at end
			savepath = os.path.join(loc, "value_development.gif")
			imageio.mimsave(savepath, gif_frames, format='GIF', duration=0.25)
			self.log(f"Saved visualizations of first state values to {savepath}")
		elif not has_image_tools: self.log(f"Visualizaiton of first state values could not be saved: Install imageio and networkx to do this")
	def plot_value_targets(self, loc: str, show=False):
		self.log("Plotting average value targets")
		plt.figure(figsize=(19.2, 10.8))
		for target, rollout in zip(self.avg_value_targets, self.evaluations):
			plt.plot(self.depths, target, label=f"Rollout {rollout}")
		plt.legend(loc=1)
		plt.xlabel("Scrambling depth")
		plt.ylabel("Average target value")
		path = os.path.join(loc, "avg_target_values.png")
		plt.savefig(path)
		if show: plt.show()
		plt.clf()
		self.log(f"Saved value target plot to {path}")

	def plot_net_changes(self, loc: str, show=False):
		self.log("Plotting changes to network parameters")
		plt.figure(figsize=(19.2, 10.8))
		plt.plot(self.evaluations, np.cumsum(self.param_changes), label="Cumulative change in network parameters")
		plt.plot(self.evaluations, self.param_total_changes, linestyle="dashdot", label="Change in parameters since original network")
		plt.legend(loc=2)
		plt.xlabel(f"Rollout number")
		plt.ylabel("Euclidian distance")
		plt.grid(True)
		path = os.path.join(loc, "parameter_changes.png")
		plt.savefig(path)
		if show: plt.show()
		plt.clf()
		self.log(f"Saved network change plot to {path}")

