from collections import deque
from typing import List

import numpy as np
import torch

from librubiks.utils import TickTock

from librubiks import gpu, no_grad
from librubiks.model import Model, ModelConfig
from librubiks.cube import Cube


class Searcher:
	eps = np.finfo("float").eps
	_explored_states = 0

	def __init__(self):
		self.action_queue = deque()
		self.tt = TickTock()

	@no_grad
	def search(self, state: np.ndarray, time_limit: float=None, max_states: int=None) -> bool:
		# Returns whether a path was found and generates action queue
		# Implement _step method for searchers that look one step ahead, otherwise overwrite this method
		self.reset()
		self.tt.tick()
		assert time_limit or max_states
		time_limit = time_limit or 1e10
		max_states = max_states or int(1e10)

		if Cube.is_solved(state): return True
		while self.tt.tock() < time_limit and len(self) < max_states:
			action, state, solution_found = self._step(state)
			self.action_queue.append(action)
			if solution_found:
				self._explored_states = len(self.action_queue)
				return True

		self._explored_states = len(self.action_queue)
		return False

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		"""
		Takes a step given a stae
		:param state: numpy array containing a state
		:return: Action index, new state, is solved
		"""
		raise NotImplementedError

	def reset(self):
		self._explored_states = 0
		self.action_queue = deque()
		self.tt.reset()
		if hasattr(self, "net"):
			self.net.eval()

	def __str__(self):
		raise NotImplementedError

	def __len__(self):
		# Returns number of states explored
		return self._explored_states


class DeepSearcher(Searcher):
	def __init__(self, net: Model):
		super().__init__()
		self.net = net

	@classmethod
	def from_saved(cls, loc: str, use_best: bool):
		net = Model.load(loc, load_best=use_best)
		net.to(gpu)
		return cls(net)

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		raise NotImplementedError


class RandomDFS(Searcher):
	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		action = np.random.randint(Cube.action_dim)
		state = Cube.rotate(state, *Cube.action_space[action])
		return action, state, Cube.is_solved(state)

	def __str__(self):
		return "Random depth-first search"


class BFS(Searcher):

	states = dict()

	def search(self, state: np.ndarray, time_limit: float=None, max_states: int=None) -> (np.ndarray, bool):
		self.reset()
		self.tt.tick()
		assert time_limit or max_states
		time_limit = time_limit or 1e10
		max_states = max_states or int(1e10)

		if Cube.is_solved(state): return True

		# Each element contains the state from which it came and the action taken to get to it
		self.states = { state.tostring(): (None, None) }
		queue = deque([state])
		while self.tt.tock() < time_limit and len(self) < max_states:
			state = queue.popleft()
			tstate = state.tostring()
			for i, action in enumerate(Cube.action_space):
				new_state = Cube.rotate(state, *action)
				new_tstate = new_state.tostring()
				if new_tstate in self.states:
					continue
				elif Cube.is_solved(new_state):
					self.action_queue.appendleft(i)
					while self.states[tstate][0] is not None:
						self.action_queue.appendleft(self.states[tstate][1])
						tstate = self.states[tstate][0]
					return True
				else:
					self.states[new_tstate] = (tstate, i)
					queue.append(new_state)

		return False

	def __str__(self):
		return "Breadth-first search"
	
	def __len__(self):
		return len(self.states)


class PolicySearch(DeepSearcher):

	def __init__(self, net: Model, sample_policy=False):
		super().__init__(net)
		self.sample_policy = sample_policy

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		policy = torch.nn.functional.softmax(self.net(Cube.as_oh(state), value=False).cpu(), dim=1).numpy().squeeze()
		action = np.random.choice(Cube.action_dim, p=policy) if self.sample_policy else policy.argmax()
		state = Cube.rotate(state, *Cube.action_space[action])
		return action, state, Cube.is_solved(state)

	@classmethod
	def from_saved(cls, loc: str, use_best: bool, sample_policy=False):
		net = Model.load(loc, load_best=use_best)
		net.to(gpu)
		return cls(net, sample_policy)

	def __str__(self):
		return f"{'Sampled' if self.sample_policy else 'Greedy'} policy"


class ValueSearch(DeepSearcher):

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		substates = Cube.multi_rotate(Cube.repeat_state(state, Cube.action_dim), *Cube.iter_actions())
		solutions = Cube.multi_is_solved(substates)
		if np.any(solutions):
			action = np.where(solutions)[0][0]
			return action, substates[action], True
		else:
			substates_oh = Cube.as_oh(substates)
			v = self.net(substates_oh, policy=False).squeeze().cpu().numpy()
			action = np.argmax(v)
			return action, substates[action], False

	def __str__(self):
		return "Greedy value"


class MCTS(DeepSearcher):

	_expand_nodes = 1000  # Expands stack by 1000, then 2000, then 4000 and etc. each expansion
	n_states = 0
	indices = dict()  # Key is state.tostring(). Contains index of state in the next arrays. Index 0 is not used
	states: np.ndarray
	neighbors: np.ndarray  # n x 12 array of neighbor indices. As first index is unused, np.all(self.neighbors, axis=1) can be used
	leaves: np.ndarray  # Boolean vector containing whether a node is a leaf
	P: np.ndarray
	V: np.ndarray
	N: np.ndarray
	W: np.ndarray
	L: np.ndarray

	def __init__(self, net: Model, c: float, nu: float, search_graph: bool, workers: int, policy_type: str):
		super().__init__(net)
		self.c = c
		self.nu = nu
		self.search_graph = search_graph
		self.workers = workers
		self.policy_type = policy_type

		self.expand_nodes = 1000

	def reset(self):
		super().reset()
		self.indices   = dict()
		self.states    = np.empty((self.expand_nodes, *Cube.shape()), dtype=Cube.dtype)
		self.neighbors = np.zeros((self.expand_nodes, Cube.action_dim), dtype=int)
		self.leaves    = np.ones(self.expand_nodes, dtype=bool)
		self.P         = np.empty((self.expand_nodes, Cube.action_dim))
		self.V         = np.empty(self.expand_nodes)
		self.N         = np.zeros((self.expand_nodes, Cube.action_dim), dtype=int)
		self.W         = np.zeros((self.expand_nodes, Cube.action_dim))
		self.L         = np.zeros((self.expand_nodes, Cube.action_dim))

	def increase_stack_size(self):
		expand_size    = len(self.states)
		self.states	   = np.concatenate([self.states, np.empty((expand_size, *Cube.shape()), dtype=Cube.dtype)])
		self.neighbors = np.concatenate([self.neighbors, np.zeros((expand_size, Cube.action_dim), dtype=int)])
		self.leaves    = np.concatenate([self.leaves, np.ones(expand_size, dtype=bool)])
		self.P         = np.concatenate([self.P, np.empty((expand_size, Cube.action_dim))])
		self.V         = np.concatenate([self.V, np.empty(expand_size)])
		self.N         = np.concatenate([self.N, np.zeros((expand_size, Cube.action_dim), dtype=int)])
		self.W         = np.concatenate([self.W, np.zeros((expand_size, Cube.action_dim))])
		self.L         = np.concatenate([self.L, np.zeros((expand_size, Cube.action_dim))])

	@no_grad
	def search(self, state: np.ndarray, time_limit: float=None, max_states: int=None) -> bool:
		self.reset()
		self.tt.tick()
		assert time_limit or max_states
		time_limit = time_limit or 1e10
		max_states = max_states or int(1e10)

		self.indices[state.tostring()] = 1
		self.states[1] = state
		if Cube.is_solved(state): return True
		oh = Cube.as_oh(state)
		p, v = self.net(oh)
		self.P[1] = p.softmax(dim=1).cpu().numpy()
		self.V[1] = v.cpu().numpy()

		paths = [deque()]
		leaves = np.array([1], dtype=int)
		workers = 1
		while self.tt.tock() < time_limit and len(self) + self.workers * Cube.action_dim <= max_states:
			self.tt.profile("Expanding leaves")
			solve_leaf, solve_action = self.expand_leaves(np.array(leaves))
			self.tt.end_profile("Expanding leaves")

			# If a solution is found
			if solve_leaf != -1:
				self.action_queue = paths[solve_leaf] + deque([solve_action])
				if self.search_graph:
					leaf_state = self.states[leaves[solve_leaf]]
					solved_state = Cube.rotate(leaf_state, *Cube.action_space[solve_action])
					self._complete_graph()
					self._shorten_action_queue(self.indices[solved_state.tostring()])
				return True

			# Find leaves
			paths, leaves = zip(*[self.find_leaf(time_limit) for _ in range(workers)])
			workers = min(workers+1, self.workers)

		return False

	def expand_leaves(self, leaves_idcs: np.ndarray) -> (int, int):
		"""
		Expands all given states which are given by the indices in leaves_idcs
		Returns the index of the leaf and the action to solve it
		Both are -1 if no solution is found
		"""

		leaf_idx, action_idx = -1, -1

		# Ensure space in stacks
		if len(self) + len(leaves_idcs) * Cube.action_dim + 1 > len(self.states):
			self.increase_stack_size()

		# Explore new states
		self.tt.profile("Get substates")
		states = self.states[leaves_idcs]
		substates = Cube.multi_rotate(np.repeat(states, Cube.action_dim, axis=0), *Cube.iter_actions(len(states)))
		self.tt.end_profile("Get substates")

		actions_taken = np.tile(np.arange(Cube.action_dim), len(leaves_idcs))
		repeated_leaves_idcs = np.repeat(leaves_idcs, Cube.action_dim)

		# Check for solution and return if found
		self.tt.profile("Check for solved state")
		solved_new_states = Cube.multi_is_solved(substates)
		solved_new_states_idcs = np.where(solved_new_states)[0]
		if solved_new_states_idcs.size:
			i = solved_new_states_idcs[0]
			leaf_idx, action_idx = i // Cube.action_dim, actions_taken[i]
		self.tt.end_profile("Check for solved state")

		substate_strs		= [s.tostring() for s in substates]
		get_substate_strs	= lambda bools: [s for s, b in zip(substate_strs, bools) if b]  # Alternative to boolean list indexing

		self.tt.profile("Classify new/old substates")
		seen_substates		= np.array([s in self.indices for s in substate_strs])  # Boolean array: Substates that have been seen before
		unseen_substates	= ~seen_substates  # Boolean array: Substates that have not been seen before
		self.tt.end_profile("Classify new/old substates")

		self.tt.profile("Handle duplicates")
		last_occurences		= np.array([s not in substate_strs[i+1:] for i, s in enumerate(substate_strs)])  # To prevent duplicates. O(n**2) goes brrrr
		last_seen			= last_occurences & seen_substates  # Boolean array: Last occurances of substates that have been seen before
		last_unseen			= last_occurences & unseen_substates  # Boolean array: Last occurances of substates that have not been seen before
		self.tt.end_profile("Handle duplicates")

		self.tt.profile("Update indices")
		new_states			= substates[last_unseen]  # Substates that are not already in the graph. Without duplicates
		new_states_idcs		= len(self) + np.arange(last_unseen.sum()) + 1  # Indices in self.states corresponding to new_states
		new_idcs_dict		= { s: i for i, s in zip(new_states_idcs, get_substate_strs(last_unseen)) }
		self.indices.update(new_idcs_dict)
		substate_idcs		= np.array([self.indices[s] for s in substate_strs])
		old_states_idcs		= substate_idcs[last_seen]  # Indices in self.states corresponding to old_states
		self.tt.end_profile("Update indices")

		# Update states and neighbors
		self.states[new_states_idcs] = substates[last_unseen]
		self.neighbors[repeated_leaves_idcs, actions_taken] = substate_idcs
		self.neighbors[substate_idcs, Cube.rev_actions(actions_taken)] = repeated_leaves_idcs

		self.tt.profile("One-hot encoding")
		new_states_oh = Cube.as_oh(new_states)
		self.tt.end_profile("One-hot encoding")
		self.tt.profile("Feedforward")
		if self.policy_type == "p":
			p, v = self.net(new_states_oh)
			p, v = p.softmax(dim=1).cpu().numpy(), v.squeeze().cpu().numpy()
			self.P[new_states_idcs] = p
		else:
			v = self.net(new_states_oh, policy=False)
			v = v.squeeze().cpu().numpy()
		# Updates all values for new states
		self.V[new_states_idcs] = v
		self.tt.end_profile("Feedforward")

		# Updates leaves
		self.leaves[leaves_idcs] = False
		self.leaves[old_states_idcs[self.neighbors[old_states_idcs].all(axis=1)]] = False

		self.tt.profile("Update W")
		neighbor_idcs = self.neighbors[leaves_idcs].ravel()
		values = self.V[neighbor_idcs].reshape((len(leaves_idcs), Cube.action_dim))
		Ws = values.max(axis=1)
		self.W[neighbor_idcs, Cube.rev_actions(actions_taken)] = np.repeat(Ws, Cube.action_dim)
		self.tt.end_profile("Update W")

		softmax = lambda x: (np.exp(x).T / np.exp(x).sum(axis=1)).T
		if self.policy_type == "v":
			self.P[leaves_idcs] = softmax(values)
		elif self.policy_type == "w":
			Ws = self.W[leaves_idcs].reshape((len(leaves_idcs), Cube.action_dim))
			self.P[leaves_idcs] = softmax(Ws)
		
		self.L[...] = 0

		return leaf_idx, action_idx

	def find_leaf(self, time_limit: float) -> (deque, int):
		"""
		Searches the tree starting from starting state using self.workers workers
		Returns a list of paths and an array containing indices of leaves
		"""
		path = deque()
		current_index = 1
		self.tt.profile("Exploring next node")
		while not self.leaves[current_index] and self.tt.tock() < time_limit:
			sqrtN = np.sqrt(self.N[current_index].sum())
			if sqrtN < self.eps:
				# If no actions have been taken from this before
				action = np.random.randint(Cube.action_dim)
			else:
				# If actions have been taken from this state before
				U = self.c * self.P[current_index] * sqrtN / (1 + self.N[current_index])
				Q = self.W[current_index] - self.L[current_index]
				action = (U + Q).argmax()
			# Updates N and virtual loss
			self.N[current_index, action] += 1
			self.L[current_index, action] += self.nu
			path.append(action)
			current_index = self.neighbors[current_index, action]
		self.tt.end_profile("Exploring next node")
		return path, current_index

	def _complete_graph(self):
		"""
		Ensures that the graph is complete by expanding around all leaves and updating neighbors
		"""
		self.tt.profile("Complete graph")
		leaves_idcs = np.where(self.leaves[:len(self)+1])[0][1:]
		actions_taken = np.tile(np.arange(Cube.action_dim), len(leaves_idcs))
		repeated_leaves_idcs = np.repeat(leaves_idcs, Cube.action_dim)
		substates = Cube.multi_rotate(self.states[repeated_leaves_idcs], *Cube.iter_actions(len(leaves_idcs)))
		substate_strs = [s.tostring() for s in substates]
		substate_idcs = np.array([self.indices[s] if s in self.indices else 0 for s in substate_strs])
		self.neighbors[repeated_leaves_idcs, actions_taken] = substate_idcs
		self.neighbors[substate_idcs, Cube.rev_actions(actions_taken)] = repeated_leaves_idcs
		self.tt.end_profile("Complete graph")

	def _shorten_action_queue(self, solved_index: int):
		if solved_index == 1: return
		self.action_queue = deque()
		visited = {1: (None, None)}  # Contains indices that have been visited
		q = deque([1])
		while q:
			v = q.popleft()
			for i, n in enumerate(self.neighbors[v]):
				if not n or n in visited:
					continue
				elif n == solved_index:
					self.action_queue.appendleft(i)
					while visited[v][0] is not None:
						self.action_queue.appendleft(visited[v][1])
						v = visited[v][0]
					return
				else:
					visited[n] = (v, i)
					q.append(n)

	@classmethod
	def from_saved(cls, loc: str, use_best: bool, c: float, nu: float, search_graph: bool, workers: int, policy_type: str):
		net = Model.load(loc, load_best=use_best)
		net.to(gpu)
		return cls(net, c=c, nu=nu, search_graph=search_graph, workers=workers, policy_type=policy_type)

	def __str__(self):
		return ("BFS" if self.search_graph else "Naive") + f"MCTS (c={self.c})"

	def __len__(self):
		return len(self.indices)


class MCTS_(DeepSearcher):

	_expand_nodes = 1000  # Expands stack by 1000, then 2000, then 4000 and etc. each expansion
	n_states = 0
	indices = dict()  # Key is state.tostring(). Contains index of state in the next arrays. Index 0 is not used
	states: np.ndarray
	neighbors: np.ndarray  # n x 12 array of neighbor indices. As first index is unused, np.all(self.neighbors, axis=1) can be used
	leaves: np.ndarray  # Boolean vector containing whether a node is a leaf
	P: np.ndarray
	V: np.ndarray
	N: np.ndarray
	W: np.ndarray
	L: np.ndarray

	def __init__(self, net: Model, c: float, nu: float, search_graph: bool, workers: int, policy_type: str):
		super().__init__(net)
		self.c = c
		self.nu = nu
		self.search_graph = search_graph
		self.workers = workers
		self.policy_type = policy_type

		self.expand_nodes = 1000

	def reset(self):
		super().reset()
		self.indices   = dict()
		self.states    = np.empty((self.expand_nodes, *Cube.shape()), dtype=Cube.dtype)
		self.neighbors = np.zeros((self.expand_nodes, Cube.action_dim), dtype=int)
		self.leaves    = np.ones(self.expand_nodes, dtype=bool)
		self.P         = np.empty((self.expand_nodes, Cube.action_dim))
		self.V         = np.empty(self.expand_nodes)
		self.N         = np.zeros((self.expand_nodes, Cube.action_dim), dtype=int)
		self.W         = np.zeros((self.expand_nodes, Cube.action_dim))
		self.L         = np.zeros((self.expand_nodes, Cube.action_dim))

	def increase_stack_size(self):
		expand_size    = len(self.states)
		self.states	   = np.concatenate([self.states, np.empty((expand_size, *Cube.shape()), dtype=Cube.dtype)])
		self.neighbors = np.concatenate([self.neighbors, np.zeros((expand_size, Cube.action_dim), dtype=int)])
		self.leaves    = np.concatenate([self.leaves, np.ones(expand_size, dtype=bool)])
		self.P         = np.concatenate([self.P, np.empty((expand_size, Cube.action_dim))])
		self.V         = np.concatenate([self.V, np.empty(expand_size)])
		self.N         = np.concatenate([self.N, np.zeros((expand_size, Cube.action_dim), dtype=int)])
		self.W         = np.concatenate([self.W, np.zeros((expand_size, Cube.action_dim))])
		self.L         = np.concatenate([self.L, np.zeros((expand_size, Cube.action_dim))])

	@no_grad
	def search(self, state: np.ndarray, time_limit: float=None, max_states: int=None) -> bool:
		self.reset()
		self.tt.tick()
		assert time_limit or max_states
		time_limit = time_limit or 1e10
		max_states = max_states or int(1e10)

		self.indices[state.tostring()] = 1
		self.states[1] = state
		if Cube.is_solved(state): return True

		path = [1]
		actions_taken = []
		while self.tt.tock() < time_limit and len(self) + Cube.action_dim <= max_states:
			self.tt.profile("Expanding leaves")
			solve_leaf, solve_action = self.expand_leaf(path, actions_taken)
			self.tt.end_profile("Expanding leaves")

			# If a solution is found
			if solve_leaf:
				self.action_queue = path + [solve_action]
				if self.search_graph:
					self._complete_graph()
					self._shorten_action_queue(solve_leaf)
				return True

			# Find leaves
			path, actions_taken = self.find_leaf(time_limit)

		self.action_queue = path  # Generates a best guess action queue in case of no

		return False

	def expand_leaf(self, visited_states_idcs: list, actions_taken: list) -> (int, int):
		"""
		Expands around the given leaf and updates V and W in all visited_states_idcs
		Returns the index of the leaf and the action to solve it
		Both are 0 if no solution is found
		:param visited_states_idcs: List of states that have been visited including the starting node. Length n
		:param actions_taken: List of actions taken from starting state. Length n-1
		:return: The index of the leaf that is the solution and the action that must be taken from leaf_index.
			Both are 0 if solution is not found
		"""
		if len(self) + Cube.action_dim > len(self.states):
			self.increase_stack_size()

		leaf_index = visited_states_idcs[-1]

		self.tt.profile("Get substates")
		state = self.states[leaf_index]
		substates = Cube.multi_rotate(Cube.repeat_state(state), *Cube.iter_actions())
		self.tt.end_profile("Get substates")

		# Check what states have been seen already
		substate_strs = [s.tostring() for s in substates]  # Unique identifier for each substate
		get_substate_strs = lambda bools: [s for s, b in zip(substate_strs, bools) if b]
		seen_substates = np.array([s in self.indices for s in substate_strs])  # States already in the graph
		unseen_substates = ~seen_substates  # States not already in the graph

		self.tt.profile("Update indices and states")
		new_states_idcs = len(self) + np.arange(unseen_substates.sum()) + 1
		new_idcs_dict = { s: i for i, s in zip(new_states_idcs, get_substate_strs(unseen_substates)) }
		self.indices.update(new_idcs_dict)
		substate_idcs = np.array([self.indices[s] for s in substate_strs])
		self.states[substate_idcs] = substates
		self.tt.end_profile("Update indices and states")

		self.tt.profile("Check for solution")
		solved_substate = np.where(Cube.multi_is_solved(substates))[0]
		if solved_substate.size:
			action = solved_substate[0]
			new_index = len(self.indices) + 1
			self.indices[substate_strs[action]] = new_index
			self.states[new_index] = substates[action]
			self.neighbors[leaf_index, action] = new_index
			self.neighbors[new_index, Cube.rev_action(action)] = leaf_index
			return leaf_index, action
		self.tt.end_profile("Check for solution")

		# Update policy, value, and W
		self.tt.profile("One-hot encoding")
		state_oh = Cube.as_oh(state)  # TODO: Consider computing values of substates instead. May save time. Will be necessary with value, if softmax(V/W) is used
		self.tt.end_profile("One-hot encoding")
		self.tt.profile("Feedforward")
		p, v = self.net(state_oh)
		p, v = p.cpu().softmax(dim=1).numpy().squeeze(), float(v.cpu().squeeze())
		self.tt.end_profile("Feedforward")
		self.tt.profile("Update P, V, and W")
		self.P[leaf_index] = p
		existing_values = self.V[visited_states_idcs[:-1]]  # TODO: What do they mean by 'backing up the value'?
		self.V[visited_states_idcs] = max(existing_values.max() if existing_values.size else v, v)
		W = np.vstack([self.W[visited_states_idcs[:-1], actions_taken], np.ones(len(actions_taken))*self.V[leaf_index]])
		self.W[visited_states_idcs[:-1], actions_taken] = W.max(axis=0)
		self.tt.end_profile("Update P, V, and W")

		self.tt.profile("Update neigbors and leaf status")
		actions = np.arange(Cube.action_dim)
		self.neighbors[leaf_index, actions] = substate_idcs
		self.neighbors[substate_idcs, Cube.rev_actions(actions)] = leaf_index
		self.leaves[leaf_index] = False
		self.tt.end_profile("Update neigbors and leaf status")

		# Update N and L
		self.tt.profile("Update N and L")
		self.N[visited_states_idcs[:-1], actions_taken] += 1
		self.L[visited_states_idcs[:-1], actions_taken] += self.nu
		self.tt.end_profile("Update N and L")

		return 0, 0

	def find_leaf(self, time_limit: float) -> (list, list):
		"""
		Searches the tree starting from starting state
		Returns a list of visited states (as indices for self.states) and a list of actions taken
		"""
		current_index = 1
		path = [1]
		actions_taken = []
		self.tt.profile("Exploring next node")
		while not self.leaves[current_index] and self.tt.tock() < time_limit:
			sqrtN = np.sqrt(self.N[current_index].sum())
			if sqrtN < self.eps:
				# If no actions have been taken from this before
				action = np.random.randint(Cube.action_dim)
			else:
				# If actions have been taken from this state before
				U = self.c * self.P[current_index] * sqrtN / (1 + self.N[current_index])
				Q = self.W[current_index] - self.L[current_index]
				action = (U + Q).argmax()
			current_index = self.neighbors[current_index, action]
			path.append(current_index)
			actions_taken.append(action)
		self.tt.end_profile("Exploring next node")
		return path, actions_taken

	def _complete_graph(self):
		"""
		Ensures that the graph is complete by expanding around all leaves and updating neighbors
		"""
		self.tt.profile("Complete graph")
		leaves_idcs = np.where(self.leaves[:len(self)+1])[0]
		actions_taken = np.tile(Cube.action_dim, len(leaves_idcs))
		repeated_leaves_idcs = np.repeat(leaves_idcs, Cube.action_dim)
		substates = Cube.multi_rotate(self.states[repeated_leaves_idcs], *Cube.iter_actions(len(leaves_idcs)))
		substate_strs = [s.tostring() for s in substates]
		substate_idcs = np.array([self.indices[s] if s in self.indices else s for s in substate_strs])
		self.neighbors[repeated_leaves_idcs, actions_taken] = substate_idcs
		self.neighbors[substate_idcs, Cube.rev_actions(actions_taken)] = repeated_leaves_idcs
		self.tt.end_profile("Complete graph")

	def _shorten_action_queue(self, start_index: int):
		visited = {start_index}  # Contains indices that have been visited
		q = deque([start_index])
		while q:
			v = q.popleft()
			for i, n in enumerate(self.neighbors[v]):
				if not n:
					continue
				elif n == 1:
					# TODO: Save paths and break here
					return
				visited.add(n)
				q.append(n)


	@classmethod
	def from_saved(cls, loc: str, use_best: bool, c: float, nu: float, search_graph: bool, workers: int, policy_type: str):
		net = Model.load(loc, load_best=use_best)
		net.to(gpu)
		return cls(net, c=c, nu=nu, search_graph=search_graph, workers=workers, policy_type=policy_type)

	def __str__(self):
		return f"MCTS {'with' if self.search_graph else 'without'} BFS (c={self.c}, nu={self.nu}, pt={self.policy_type})"

	def __len__(self):
		return len(self.indices)


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

		oh = Cube.as_oh(state)
		p, v = self.net(oh)  # Policy and value
		self.open[state.tostring()] = {'g_cost': 0, 'h_cost': -float(v.cpu()), 'parent': 'Starting node'}
		del p, v

		# explore leaves
		while self.tt.tock() < time_limit:
			if len(self.open) == 0: # FIXME: bug
				print('AStar open was empty.')
				return False
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
		node = np.fromstring(node, dtype=Cube.dtype)
		for i in range(Cube.action_dim):
			neighbor = Cube.rotate(node, *Cube.action_space[i])
			neighbors[i] = neighbor.tostring()
		return neighbors

	def get_g_cost(self, node: str):
		return self.closed[node]['g_cost'] + 1

	def get_h_cost(self, node: str):
		node = np.fromstring(node, dtype=Cube.dtype)
		if Cube.is_solved(node):
			return 0
		else:
			oh = Cube.as_oh(node)
			p, v = self.net(oh)
			return -float(v.cpu())

	def __str__(self):
		return f"AStar search"

	def __len__(self):
		node = list(self.closed)[-1]
		count = 0
		while True:
			node = self.closed[node]['parent']
			if node == 'Starting node': return count
			count += 1


class DankSearch(DeepSearcher):

	def __init__(self, net, epsilon: float, workers: int, depth: int):
		super().__init__(net)
		self.epsilon = epsilon
		self.workers = workers
		self.depth = depth

	@no_grad
	def search(self, state: np.ndarray, time_limit: float=None, max_states: int=None) -> bool:
		self.reset()
		self.tt.tick()
		assert time_limit or max_states
		time_limit = time_limit or 1e10
		max_states = max_states or int(1e10)

		if Cube.is_solved(state):
			return True

		states = Cube.repeat_state(state, self.workers)
		while self.tt.tock() < time_limit and len(self) + self.workers * self.depth <= max_states:
			paths, states, states_oh, solved = self.expand(states)
			if solved != (-1, -1):
				self.action_queue += deque(paths[solved[0]][:solved[1]])
				return True
			assert torch.all(Cube.as_oh(states)==states_oh)
			v = self.net(states_oh, policy=False).cpu().squeeze()
			best_value_state_index = int(v.argmax())
			worker, depth = best_value_state_index // self.depth, best_value_state_index % self.depth
			self.action_queue += deque(paths[worker][:depth])  # TODO: Problably bug with depth = 0

		return False

	def _get_indices(self, depth: int) -> np.ndarray:
		return np.arange(self.workers) * self.depth + depth

	def expand(self, states: np.ndarray) -> (list, np.ndarray, torch.tensor, tuple):
		paths = [[] for _ in range(self.workers)]
		new_states = np.empty((self.workers * self.depth, *Cube.shape()), dtype=Cube.dtype)
		new_states_oh = torch.empty(self.workers * self.depth, Cube.get_oh_shape(), dtype=torch.float)
		for d in range(self.depth):
			use_random = np.random.choice(2, self.workers, p=[1-self.epsilon, self.epsilon]).astype(bool)
			use_policy = ~use_random
			actions = np.empty(self.workers, dtype=int)
			actions[use_random] = np.random.randint(0, Cube.action_dim, use_random.sum())
			states_oh = Cube.as_oh(states[use_policy])
			p = self.net(states_oh, value=False).cpu().numpy()
			actions[use_policy] = p.argmax(axis=1)
			[path.append(a) for path, a in zip(paths, actions)]

			faces, dirs = actions % 6, actions // 6
			states = Cube.multi_rotate(states, faces, dirs)
			solved_states = Cube.multi_is_solved(states)
			if np.any(solved_states):
				w = np.where(solved_states)[0][0]
				return paths, None, None, (w, d)
			new_states[self._get_indices(d)] = states
			new_states_oh[self._get_indices(d)] = Cube.as_oh(states)  # Don't waste time like this
		self._explored_states += len(new_states)

		return paths, new_states, new_states_oh, (-1, -1)

