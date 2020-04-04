from collections import deque
from copy import deepcopy

import numpy as np
import torch

from src.rubiks.model import Model
from src.rubiks.cube.cube import Cube
from src.rubiks.utils import cpu, gpu
from src.rubiks.utils.ticktock import TickTock

class Node:
	def __init__(self, state: np.ndarray, policy: np.ndarray, value: float, from_node=None, action_idx: int=None):
		self.is_leaf = True
		self.state = state
		self.P = policy
		self.value = value
		# self.neighs[i] is a tuple containing the state obtained by the action Cube.action_space[i]
		# Tuples are used, so they can be used for lookups
		self.neighs = [None] * 12
		self.N = np.zeros(12)
		self.W = np.empty(12)
		self.L = np.zeros(12)
		if action_idx is not None:
			from_action_idx = Cube.rev_action(action_idx)
			self.neighs[from_action_idx] = from_node
			self.W[from_action_idx] = from_node.value

	def __str__(self):
		return "\n".join([
			"----- Node -----",
			f"Leaf:      {self.is_leaf}",
			f"State:     {tuple(self.state)}",
			f"N:         {self.N}",
			f"W:         {self.W}",
			f"Neighbors: {[id(x) if x is not None else None for x in self.neighs]}",
			"----------------",
		])



class Searcher:
	def __init__(self):
		self.action_queue = deque()

	def search(self, state: np.ndarray, time_limit: int) -> bool:
		# Returns whether a path was found
		raise NotImplementedError

class RandomDFS(Searcher):
	def search(self, state: np.ndarray, time_limit: int):
		tt = TickTock()
		tt.tick()
		while tt.tock() < time_limit:
			action = np.random.randint(Cube.action_dim)
			state = Cube.rotate(state, *Cube.action_space[action])

			self.action_queue.append(action)
			if Cube.is_solved(state): return True
		return False

class BFS(Searcher):
	def search(self, state: np.ndarray, time_limit: int):
		raise NotImplementedError

class MCTS(Searcher):
	def __init__(self, net: Model, c: float=1, nu: float=0):
		super().__init__()
		self.states = dict()
		self.net = net
		self.c = c
		self.nu = nu


	def search(self, state: np.ndarray, time_limit: int):
		if Cube.is_solved(state):
			return deque()
		tt = TickTock()
		tt.tick()
		oh = Cube.as_oh(state).to(gpu)
		with torch.no_grad():
			p, v = self.net(oh)
		self.states[tuple(state)] = Node(state, p.cpu().numpy().ravel(), float(v.cpu()))
		del p, v
		solve_action = self.expand_leaf(self.states[tuple(state)])
		if solve_action != -1:
			self.action_queue = deque([solve_action])
			return True
		while tt.tock() < time_limit:
			path, leaf = self.search_leaf(self.states[tuple(state)])
			solve_action = self.expand_leaf(leaf)
			if solve_action != -1:
				self.action_queue = path + deque([solve_action])
				return True
		return bool(self.action_queue)

	def search_leaf(self, node: Node) -> (list, Node):
		# Finds leaf starting from state
		path = deque()
		while not node.is_leaf:
			U = self.c * node.P * np.sqrt(node.N.sum()) / (1 + node.N)
			Q = node.W - node.L
			action = np.argmax(U + Q)
			node.N[action] += 1
			node.L[action] += self.nu
			path.append(action)
			node = node.neighs[action]
		return path, node

	def expand_leaf(self, leaf: Node) -> int:
		# Expands at leaf node and check if solved state in new states
		# Return -1 if no action gives solved state else action index
		# print(f"----- Expanding leaf {tuple(leaf.state)} -----")
		no_neighs = np.array([i for i in range(12) if leaf.neighs[i] is None])  # Neighbors that have to be expanded to
		unknown_neighs = list(np.arange(len(no_neighs)))  # Some unknown neighbors may already be known but just not connected
		new_states = np.empty((len(no_neighs), *Cube.get_solved_instance().shape), dtype=Cube.dtype)
		for i in reversed(range(len(no_neighs))):
			action = no_neighs[i]
			new_states[i] = Cube.rotate(leaf.state, *Cube.action_space[action])
			if Cube.is_solved(new_states[i]):
				# print("SOLVED"*10)
				return action
			# If new leaf state is already known, the tree is updated, and the neighbor is no longer considered
			tstate = tuple(new_states[i])
			if tstate in self.states:
				leaf.neighs[action] = self.states[tstate]
				self.states[tstate].neighs[Cube.rev_action(action)] = leaf
				unknown_neighs.pop(i)
		no_neighs = no_neighs[unknown_neighs]
		new_states = new_states[unknown_neighs]

		new_states_oh = torch.empty(len(unknown_neighs), Cube.get_oh_shape())
		# Passes new states through net
		for i in range(len(no_neighs)):
			new_states_oh[i] = Cube.as_oh(new_states[i])
		new_states_oh = new_states_oh.to(gpu)
		with torch.no_grad():
			p, v = self.net(new_states_oh)
			p, v = p.cpu().numpy(), v.cpu().numpy()
		# Generates new states
		for i, action in enumerate(no_neighs):
			new_leaf = Node(new_states[i], p[i], v[i], leaf, action)
			leaf.neighs[action] = new_leaf
			self.states[tuple(new_states[i])] = new_leaf
		if any([x is None for x in leaf.neighs]): breakpoint()
		leaf.is_leaf = False

		return -1



