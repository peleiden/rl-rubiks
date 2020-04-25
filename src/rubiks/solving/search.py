from collections import deque
from typing import List

import numpy as np
import torch

from src.rubiks import cpu, gpu, no_grad
from src.rubiks.model import Model
from src.rubiks.cube.cube import Cube
from src.rubiks.utils.ticktock import TickTock


class Node:
	def __init__(self, state: np.ndarray, policy: np.ndarray, value: float, from_node=None, action_idx: int=None):
		self.is_leaf = True  # When initiated, the node is leaf of search graph
		self.state = state
		self.P = policy
		self.value = value
		# self.neighs[i] is a tuple containing the state obtained by the action Cube.action_space[i]
		# strings are used, so they can be used for lookups
		self.neighs = [None] * Cube.action_dim
		self.N = np.zeros(Cube.action_dim)
		self.W = np.zeros(Cube.action_dim)
		self.L = np.zeros(Cube.action_dim)
		if action_idx is not None:
			from_action_idx = Cube.rev_action(action_idx)
			self.neighs[from_action_idx] = from_node

	def __str__(self):
		return "\n".join([
			"----- Node -----",
			f"Leaf:      {self.is_leaf}",
			f"State:     {tuple(self.state)}",
			f"Value:     {self.value}",
			f"N:         {self.N}",
			f"W:         {self.W}",
			f"Neighbors: {[id(x) if x is not None else None for x in self.neighs]}",
			"----------------",
		])


class Searcher:
	with_mt = False
	eps = np.finfo("float").eps

	def __init__(self):
		self.action_queue = deque()
		self.tt = TickTock()

	@no_grad
	def search(self, state: np.ndarray, time_limit: float) -> bool:
		# Returns whether a path was found and generates action queue
		# Implement _step method for searchers that look one step ahead, otherwise overwrite this method
		self.reset()
		self.tt.tick()
		if Cube.is_solved(state): return True
		while self.tt.tock() < time_limit:
			action, state, solution_found = self._step(state)
			self.action_queue.append(action)
			if solution_found: return True
		return False

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		raise NotImplementedError

	def reset(self):
		self.action_queue = deque()
		self.tt.reset()

	def __str__(self):
		raise NotImplementedError

	def __len__(self):
		# Returns number of states explored
		return len(self.action_queue)


class DeepSearcher(Searcher):
	def __init__(self, net: Model):
		super().__init__()
		self.net = net

	@classmethod
	def from_saved(cls, loc: str):
		net = Model.load(loc)
		net.to(gpu)
		return cls(net)

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		raise NotImplementedError


class RandomDFS(Searcher):
	with_mt = True  # TODO: Implement multithreading natively in search method and set to False
	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		action = np.random.randint(Cube.action_dim)
		state = Cube.rotate(state, *Cube.action_space[action])
		return action, state, Cube.is_solved(state)

	def __str__(self):
		return "Random depth-first search"

class BFS(Searcher):
	def search(self, state: np.ndarray, time_limit: float) -> (np.ndarray, bool):
		self.reset()
		self.tt.tick()

		if Cube.is_solved(state): return True

		# Each element contains the state from which it came and the corresponding action
		states = { state.tostring(): (None, None) }
		queue = deque([state])
		while self.tt.tock() < time_limit:
			state = queue.popleft()
			tstate = state.tostring()
			for i, action in enumerate(Cube.action_space):
				new_state = Cube.rotate(state, *action)
				new_tstate = new_state.tostring()
				if new_tstate in states:
					continue
				elif Cube.is_solved(new_state):
					self.action_queue.appendleft(i)
					while states[tstate][0] is not None:
						self.action_queue.appendleft(states[tstate][1])
						tstate = states[tstate][0]
					return True
				else:
					states[new_tstate] = (tstate, i)
					queue.append(new_state)
		return False

	def __str__(self):
		return "Breadth-first search"


class PolicySearch(DeepSearcher):
	with_mt = not torch.cuda.is_available()

	def __init__(self, net: Model, sample_policy=False):
		super().__init__(net)
		self.sample_policy = sample_policy

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		policy = torch.nn.functional.softmax(self.net(Cube.as_oh(state).to(gpu), value=False).cpu(), dim=1).numpy().squeeze()
		action = np.random.choice(Cube.action_dim, p=policy) if self.sample_policy else policy.argmax()
		state = Cube.rotate(state, *Cube.action_space[action])
		return action, state, Cube.is_solved(state)

	@classmethod
	def from_saved(cls, loc: str, sample_policy=False):
		net = Model.load(loc)
		net.to(gpu)
		return cls(net, sample_policy)

	def __str__(self):
		return f"Policy search {'with' if self.sample_policy else 'without'} sampling"

class MCTS(DeepSearcher):
	def __init__(self, net: Model, c: float, nu: float, search_graph: bool, workers=10):
		super().__init__(net)
		# Hyperparameters: c controls exploration and nu controls virtual loss updation us
		self.c = c
		self.nu = nu
		self.search_graph = search_graph
		self.workers = workers

		self.states = dict()
		self.net = net

	@no_grad
	def search(self, state: np.ndarray, time_limit: float) -> bool:
		self.reset()

		self.tt.tick()
		if Cube.is_solved(state): return True
		# First state is evaluated and expanded individually
		oh = Cube.as_oh(state).to(gpu)
		p, v = self.net(oh)  # Policy and value
		self.states[state.tostring()] = Node(state, p.softmax(dim=1).cpu().numpy().ravel(), float(v.cpu()))
		del p, v
		
		paths = [deque([])]
		leaves = [self.states[state.tostring()]]
		while self.tt.tock() < time_limit:
			self.tt.profile("Expanding leaves")
			solve_leaf, solve_action = self.expand_leaves(leaves)
			self.tt.end_profile("Expanding leaves")
			if solve_leaf != -1:  # If a solution is found
				self.action_queue = paths[solve_leaf] + deque([solve_action])
				if self.search_graph:
					self._shorten_action_queue()
				return True
			# Gets new paths and leaves to expand from
			paths, leaves = zip(*[self.search_leaf(self.states[state.tostring()], time_limit) for _ in range(self.workers)])

		self.action_queue = paths[solve_leaf] + deque([solve_action])  # Saves an action queue even if it loses which is its best guess
		return False

	def search_leaf(self, node: Node, time_limit: float) -> (list, Node):
		# Finds leaf starting from state
		path = deque()
		self.tt.profile("Exploring next node")
		while not node.is_leaf and self.tt.tock() < time_limit:
			sqrtN = np.sqrt(node.N.sum())
			if sqrtN < self.eps:  # Randomly chooses path the first time a state is explored
				action = np.random.choice(Cube.action_dim)
			else:
				U = self.c * node.P * sqrtN / (1 + node.N)
				Q = node.W - node.L
				g = U + Q
				action = np.argmax(g)
			node.N[action] += 1
			node.L[action] += self.nu
			path.append(action)
			node = node.neighs[action]
		self.tt.end_profile("Exploring next node")
		return path, node

	def _update_neighbors(self, state: np.ndarray):
		"""
		# Expands around state. If a new state is already in the tree, neighbor relations are updated
		# Assumes that state is already in the tree
		# Used for node expansion
		"""
		state_str = state.tostring()
		new_states = Cube.multi_rotate(
			np.tile(state, (Cube.action_dim, 1)),
			*Cube.iter_actions()
		)
		new_states_strs = [x.tostring() for x in new_states]
		for i, (new_state, new_state_str) in enumerate(zip(new_states, new_states_strs)):
			self.tt.profile("Update neighbors")
			if new_state_str in self.states:
				self.states[state_str].neighs[i] = self.states[new_state_str]
				self.states[new_state_str].neighs[Cube.rev_action(i)] = self.states[state_str]
				if all(self.states[new_state_str].neighs):
					self.states[new_state_str].is_leaf = False
			self.tt.end_profile("Update neighbors")
		if all(self.states[state_str].neighs):
			self.states[state_str].is_leaf = False

	def expand_leaves(self, leaves: List[Node]) -> (int, int):
		"""
		Expands all given leaves
		Returns the index of the leaf and the action to solve it
		Both are -1 if no solution is found
		"""

		# Explores all new states
		self.tt.profile("Getting new states to expand to")
		states = np.array([leaf.state for leaf in leaves])
		new_states = Cube.multi_rotate(np.repeat(states, Cube.action_dim, axis=0), *Cube.iter_actions(len(states)))
		self.tt.end_profile("Getting new states to expand to")

		# Checks for solutions
		self.tt.profile("Checking for solved state")
		for i, state in enumerate(new_states):
			if Cube.is_solved(state):
				leaf_idx, action_idx = i // Cube.action_dim, i % Cube.action_dim
				solved_leaf = Node(state, None, None, leaves[leaf_idx], action_idx)
				self.states[state.tostring()] = solved_leaf
				self._update_neighbors(state)
				return i // Cube.action_dim, i % Cube.action_dim
		self.tt.end_profile("Checking for solved state")

		# Gets information about new states
		new_states_str = [state.tostring() for state in new_states]
		self.tt.profile("One-hot encoding")
		new_states_oh = Cube.as_oh(new_states).to(gpu)
		self.tt.end_profile("One-hot encoding")
		self.tt.profile("Feedforward")
		policies, values = self.net(new_states_oh)
		policies, values = policies.softmax(dim=1).cpu().numpy(), values.squeeze().cpu().numpy()
		self.tt.end_profile("Feedforward")

		self.tt.profile("Generate new states")
		for i, (state, state_str, p, v) in enumerate(zip(new_states, new_states_str, policies, values)):
			leaf_idx, action_idx = i // Cube.action_dim, i % Cube.action_dim
			leaf = leaves[leaf_idx]
			if state_str not in self.states:
				new_leaf = Node(state, p, v, leaf, action_idx)
				leaf.neighs[action_idx] = new_leaf
				self.states[state_str] = new_leaf
				# It is possible to add check for existing neighbors in graph here using self._update_neighbors(state) to ensure graph completeness
				# However, this is so expensive that it has been found to reduce the number of explored states to around a quarter
				# Also, it is not a major problem, as the edges will be updated when new_leaf is expanded, so the problem only exists on the edge of the graph
				# TODO: Test performance difference after implementing this
				# TODO: Save dumb nodes when expanding. This should allow graph completeness without massive overhead
			else:
				leaf.neighs[action_idx] = self.states[state_str]
				self.states[state_str].neighs[Cube.rev_action(action_idx)] = leaf
				if all(self.states[state_str].neighs):
					self.states[state_str].is_leaf = False
			leaf.is_leaf = False
		self.tt.end_profile("Generate new states")

		self.tt.profile("Update W")
		for leaf in leaves:
			max_val = max([x.value for x in leaf.neighs])
			assert max_val != 0
			for action_idx, neighbor in enumerate(leaf.neighs):
				neighbor.W[Cube.rev_action(action_idx)] = max_val
		self.tt.end_profile("Update W")

		return -1, -1

	def expand_leaf(self, leaf: Node) -> int:
		# Expands at leaf node and checks if solved state in new states
		# Returns -1 if no action gives solved state else action index

		no_neighs = np.array([i for i in range(Cube.action_dim) if leaf.neighs[i] is None])  # Neighbors that have to be expanded to
		unknown_neighs = list(np.arange(len(no_neighs)))  # Some unknown neighbors may already be known but just not connected
		new_states = np.empty((len(no_neighs), *Cube.get_solved_instance().shape), dtype=Cube.dtype)

		self.tt.profile("Exploring child states")
		for i in reversed(range(len(no_neighs))):
			action = no_neighs[i]
			new_states[i] = Cube.rotate(leaf.state, *Cube.action_space[action])
			if Cube.is_solved(new_states[i]): return action

			# If new leaf state is already known, the tree is updated, and the neighbor is no longer considered
			state_str = new_states[i].tostring()
			if state_str in self.states:
				leaf.neighs[action] = self.states[state_str]
				self.states[state_str].neighs[Cube.rev_action(action)] = leaf
				unknown_neighs.pop(i)

		no_neighs = no_neighs[unknown_neighs]
		new_states = new_states[unknown_neighs]
		self.tt.end_profile("Exploring child states")

		# Passes new states through net
		self.tt.profile("One-hot encoding new states")
		new_states_oh = Cube.as_oh(new_states).to(gpu)
		self.tt.end_profile("One-hot encoding new states")
		self.tt.profile("Feedforwarding")
		p, v = self.net(new_states_oh)
		p, v = torch.nn.functional.softmax(p.cpu(), dim=1).cpu().numpy(), v.cpu().numpy()
		self.tt.end_profile("Feedforwarding")

		self.tt.profile("Generate new states")
		for i, action in enumerate(no_neighs):
			new_leaf = Node(new_states[i], p[i], v[i], leaf, action)
			leaf.neighs[action] = new_leaf
			self.states[new_states[i].tostring()] = new_leaf
		self.tt.end_profile("Generate new states")

		# Updates W in all non-leaf neighbors
		self.tt.profile("Update W")
		max_val = max([x.value for x in leaf.neighs])
		for action, neighbor in enumerate(leaf.neighs):
			if neighbor.is_leaf:
				continue
			neighbor.W[Cube.rev_action(action)] = max_val
		self.tt.end_profile("Update W")

		leaf.is_leaf = False
		return -1

	def _shorten_action_queue(self):
		# TODO: Implement and ensure graph completeness
		# Generates new action queue with BFS through self.states
		pass

	def reset(self):
		super().reset()
		self.states = dict()

	@classmethod
	def from_saved(cls, loc: str, c: float, nu: float, search_graph: bool, workers: int):
		net = Model.load(loc)
		net.to(gpu)
		return cls(net, c, nu, search_graph, workers)

	def __str__(self):
		return f"Monte Carlo Tree Search {'with' if self.search_graph else 'without'} graph search (c={self.c}, nu={self.nu})"
	
	def __len__(self):
		return len(self.states)


class AStar(DeepSearcher):

	def __init__(self, net: Model):
		super().__init__(net)
		self.net = net
		self.open = dict()
		self.closed = dict()

	@no_grad
	def search(self, state: np.ndarray, time_limit: float) -> bool:
		# initializing/resetting stuff
		self.tt.tick()
		self.reset()
		self.open = {}
		if Cube.is_solved(state): return True

		oh = Cube.as_oh(state).to(gpu)
		p, v = self.net(oh)  # Policy and value
		self.open[state.tostring()] = {'g_cost': 0, 'h_cost': -float(v.cpu()), 'parent': None}
		del p, v

		# explore leaves
		while self.tt.tock() < time_limit:
			# choose current node as the node in open with the lowest f cost
			idx = np.argmin([self.open[node]['g_cost'] + self.open[node]['h_cost'] for node in self.open])
			current = list(self.open)[idx]
			# add to closed and remove from open
			self.closed[current] = self.open[current]
			del self.open[current]
			if self.closed[current]['h_cost'] == 0: return True
			neighbors = self.get_neighbors(current)
			for neighbor in neighbors:
				if neighbor not in self.closed:
					g_cost = self.get_g_cost(current)
					h_cost = self.get_h_cost(neighbor)
					if neighbor not in self.open or g_cost+h_cost < self.open[neighbor]['g_cost']+self.open[neighbor]['h_cost']:
						self.open[neighbor] = {'g_cost': g_cost, 'h_cost': h_cost, 'parent': current}
		return False

	def get_neighbors(self, node: str):
		neighbors = [None] * Cube.action_dim
		node = np.fromstring(node, dtype=int)
		for i in range(Cube.action_dim):
			neighbor = Cube.rotate(node, *Cube.action_space[i])
			neighbors[i] = neighbor.tostring()
		return neighbors

	def get_g_cost(self, node: str):
		return self.closed[node]['g_cost'] + 1

	def get_h_cost(self, node: str):
		node = np.fromstring(node, dtype=int)
		if Cube.is_solved(node):
			return 0
		else:
			oh = Cube.as_oh(node).to(gpu)
			p, v = self.net(oh)
			return -float(v.cpu()) #alternativ idé: 1/float(v.cpu())

	def __str__(self):
		return f"A* Search"
	
	def __len__(self):
		# TODO
		raise NotImplementedError

