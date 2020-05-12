import os

import numpy as np
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolour

from librubiks.cube import Cube
from librubiks.model import Model
from librubiks.utils import NullLogger, Logger

colours = list(mcolour.BASE_COLORS)
tab_colours = list(mcolour.TABLEAU_COLORS)
all_colours = colours[:-1] + tab_colours[:-2]

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
		self.depths = np.arange(depth)
		extra_evals = min(evaluations[-1] if len(evaluations) else 0, extra_evals) #Wont add evals in the future (or if no evals are needed)
		self.evaluations = np.unique( np.append(evaluations, range( extra_evals )) )

		self.orig_params = None
		self.params = None

		# TODO: Implement check of first 12 substates
		# substates = Cube.sequence_scrambler(1, 2)
		# self.substate_values = list()

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
			targets = value_targets.cpu().numpy().reshape((-1, self.depth))
			self.avg_value_targets.append(targets.mean(axis=0))

			# Calculating model change
			model_change = torch.sqrt((net.get_params()-self.params)**2).mean().cpu()
			model_total_change = torch.sqrt((net.get_params()-self.orig_params)**2).mean().cpu()
			self.params = net.get_params()
			self.param_changes.append(float(model_change))
			self.param_total_changes.append(model_total_change)

			#TODO: Calculate value given to first 12 substates

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

	# def visualize_substates(self, loc: str, show=False)
		# TODO: Visualize value of substates (over time?)
		# Plot? Something cool? Graphviz?
		# pass

	def _get_evaluations_for_value(self):
		"""
		Returns a boolean vector of length len(self.evaluations) containing whether or not the curve should be in focus
		"""
		early_rollouts = 5
		late_rollouts = 10
		early_indices = np.arange(early_rollouts) * 3
		late_indices = np.unique(np.linspace(early_indices[-1], len(self.evaluations)-1, late_rollouts+1)[1:].astype(int))
		focus_rollouts = np.zeros(len(self.evaluations), dtype=bool)
		focus_rollouts[early_indices] = True
		focus_rollouts[late_indices] = True
		return focus_rollouts


	def plot_value_targets(self, loc: str, show=False):
		self.log("Plotting average value targets")
		plt.figure(figsize=(19.2, 10.8))
		focus_rollouts = self._get_evaluations_for_value()
		colours = iter(all_colours)
		filter_by_bools = lambda list_, bools: [x for x, b in zip(list_, bools) if b]
		for target, rollout in zip(filter_by_bools(self.avg_value_targets, ~focus_rollouts), filter_by_bools(self.evaluations, ~focus_rollouts)):
			plt.plot(self.depths, target, "--", color="grey", alpha=.4)
		for target, rollout in zip(filter_by_bools(self.avg_value_targets, focus_rollouts), filter_by_bools(self.evaluations, focus_rollouts)):
			plt.plot(self.depths, target, linewidth=3, color=next(colours), label=f"{rollout} Rollouts")
		plt.legend(loc=1)
		plt.xlim([0, self.depths[-1]+1])
		plt.xlabel("Scrambling depth")
		plt.ylabel("Average target value")
		path = os.path.join(loc, "avg_target_values.png")
		plt.grid(True)
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

