import heapq
from collections import deque
from typing import List

import numpy as np
import torch

from librubiks.utils import TickTock

from librubiks import gpu, no_grad, softmax
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
		if hasattr(self, "net"): self.net.eval()

	def action(self)->(int, bool):
		return Cube.action_space[self.action_queue.popleft()]

	def actions(self) -> (int, bool):
		while self.action_queue: yield self.action()

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

	def __init__(self, net: Model, c: float, search_graph: bool):
		super().__init__(net)
		self.c = c
		self.search_graph = search_graph
		self.nu = 100

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
		self.V[1] = v.cpu().numpy()
		self.P[1] = p.softmax(dim=1).cpu().numpy()
		indices_visited = [1]
		actions_taken = []
		while self.tt.tock() < time_limit and len(self) + Cube.action_dim <= max_states:
			self.tt.profile("Expanding leaves")
			solve_leaf_index, solve_action = self.expand_leaf(indices_visited, actions_taken)
			self.tt.end_profile("Expanding leaves")

			# If a solution is found
			if solve_leaf_index != -1:
				self.action_queue = deque(actions_taken) + deque([solve_action])
				if self.search_graph:
					self._complete_graph()
					self._shorten_action_queue(solve_leaf_index)
				return True

			# Find leaves
			indices_visited, actions_taken = self.find_leaf(time_limit)

		self.action_queue = deque(indices_visited)  # Generates a best guess action queue in case of no solution

		return False

	def expand_leaf(self, visited_states_idcs: list, actions_taken: list) -> (int, int):
		"""
		Expands around the given leaf and updates V and W in all visited_states_idcs
		Returns the action taken to solve the cube. -1 if no solution is found
		:param visited_states_idcs: List of states that have been visited including the starting node. Length n
		:param actions_taken: List of actions taken from starting state. Length n-1
		:return: The index of the leaf that is the solution and the action that must be taken from leaf_index.
			Both are 0 if solution is not found
		"""
		if len(self) + Cube.action_dim > len(self.states):
			self.increase_stack_size()

		leaf_index = visited_states_idcs[-1]
		solve_leaf, solve_action = -1, -1

		self.tt.profile("Get substates")
		state = self.states[leaf_index]
		substates = Cube.multi_rotate(Cube.repeat_state(state), *Cube.iter_actions())
		self.tt.end_profile("Get substates")

		# Check what states have been seen already
		substate_strs = [s.tostring() for s in substates]  # Unique identifier for each substate
		get_substate_strs = lambda bools: [s for s, b in zip(substate_strs, bools) if b]  # Shitty way to easily index into list with boolean array
		seen_substates = np.array([s in self.indices for s in substate_strs])  # States already in the graph
		unseen_substates = ~seen_substates  # States not already in the graph

		self.tt.profile("Update indices and states")
		new_states_idcs = len(self) + np.arange(unseen_substates.sum()) + 1
		new_idcs_dict = { s: i for i, s in zip(new_states_idcs, get_substate_strs(unseen_substates)) }
		self.indices.update(new_idcs_dict)
		substate_idcs = np.array([self.indices[s] for s in substate_strs])
		new_substate_idcs = substate_idcs[unseen_substates]
		new_substates = substates[unseen_substates]
		self.states[new_substate_idcs] = new_substates
		self.tt.end_profile("Update indices and states")

		self.tt.profile("Update neigbors and leaf status")
		actions = np.arange(Cube.action_dim)
		self.neighbors[leaf_index, actions] = substate_idcs
		self.neighbors[substate_idcs, Cube.rev_actions(actions)] = leaf_index
		self.leaves[leaf_index] = False
		self.tt.end_profile("Update neigbors and leaf status")

		self.tt.profile("Check for solution")
		solved_substate = np.where(Cube.multi_is_solved(substates))[0]
		if solved_substate.size:
			solve_action = solved_substate[0]
			solve_leaf = substate_idcs[solve_action]
		self.tt.end_profile("Check for solution")

		# Update policy, value, and W
		self.tt.profile("One-hot encoding")
		new_substates_oh = Cube.as_oh(new_substates)
		self.tt.end_profile("One-hot encoding")
		self.tt.profile("Feedforward")
		p, v = self.net(new_substates_oh)
		p, v = p.cpu().softmax(dim=1).numpy(), v.cpu().numpy().squeeze()
		self.tt.end_profile("Feedforward")

		self.tt.profile("Update P, V, and W")
		self.P[new_substate_idcs] = p
		self.V[new_substate_idcs] = v
		self.W[new_substate_idcs] = np.tile(v, (Cube.action_dim, 1)).T
		self.W[leaf_index] = self.V[self.neighbors[leaf_index]]
		# Data structure: First row has all existing W's and second has value of leaf that is expanded from
		W = np.vstack([self.W[visited_states_idcs[:-1], actions_taken], np.repeat(self.V[leaf_index], len(visited_states_idcs)-1)])
		self.W[visited_states_idcs[:-1], actions_taken] = W.max(axis=0)
		self.tt.end_profile("Update P, V, and W")

		# Update N and L
		self.tt.profile("Update N and L")
		if actions_taken:  # Crashes if actions_taken is empty, which happens on the first run
			self.N[visited_states_idcs[:-1], actions_taken] += 1
			self.L[visited_states_idcs[:-1], actions_taken] = 0
			self.L[visited_states_idcs[1:], Cube.rev_actions(np.array(actions_taken))] = 0
		self.tt.end_profile("Update N and L")

		return solve_leaf, solve_action

	def find_leaf(self, time_limit: float) -> (list, list):
		"""
		Searches the tree starting from starting state
		Returns a list of visited states (as indices for self.states) and a list of actions taken
		"""
		current_index = 1
		indices_visited = [current_index]
		actions_taken = []
		self.tt.profile("Exploring next node")
		while not self.leaves[current_index] and self.tt.tock() < time_limit:
			sqrtN = np.sqrt(self.N[current_index].sum())
			U = self.c * self.P[current_index] * sqrtN / (1 + self.N[current_index])
			Q = self.W[current_index] - self.L[current_index]
			action = (U + Q).argmax()
			self.L[current_index, action] += self.nu
			current_index = self.neighbors[current_index, action]
			self.L[current_index, Cube.rev_action(action)] += self.nu
			indices_visited.append(current_index)
			actions_taken.append(action)
		self.tt.end_profile("Exploring next node")
		return indices_visited, actions_taken

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
		self.tt.profile("BFS")
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
		self.tt.end_profile("BFS")

	@classmethod
	def from_saved(cls, loc: str, use_best: bool, c: float, search_graph: bool):
		net = Model.load(loc, load_best=use_best)
		net.to(gpu)
		return cls(net, c=c, search_graph=search_graph)

	def __str__(self):
		return ("BFS" if self.search_graph else "Naive") + f" MCTS (c={self.c})"

	def __len__(self):
		return len(self.indices)


class AStar(DeepSearcher):
	"""Batch Weighted A* Search
	As per Agostinelli, McAleer, Shmakov, Baldi:
	"Solving the Rubik's cube with deep reinforcement learning and search".

	Expands the `self.expansions` best nodes at a time according to cost
	f(node) = `self.lambda_` * g(node) + h(node)
	where h(node) is given as the negative value (cost-to-go) of the DNN and g(x) is the path cost
	"""

	# Queue data structure
		# Min heap. An element contains tuple of (cost, index)
		# This priority queue uses python std. lib heapq which is based on the python list.
		# We should maybe consider whether this could be done faster if we build our own implementation.
	open_queue: list

	indices = dict  # Key is state.tostring(). Contains index of state in the next arrays. Index 0 is not used
	states: np.ndarray
	parents: np.ndarray #parents[i] index of lightest parent of state i
	parent_actions: np.ndarray #parent_actions[i]: action idx taken FROM the lightest parent to state i
	G: np.ndarray
	H: np.ndarray
	closed: np.ndarray # Boolean vector. closed[i] == True iff. state i is closed (expanded)
	# Boolean vector. open_[i] == True iff. state i in open_queue closed.
	# Yes, we have two data structures for this: open_queue has sort, this has random access.
	open_: np.ndarray

	_stack_expand = 1000
		#TODO: BIG THINK
		# Do we even need to maintain a  H data structure?
		# Could we calculate cost once and save directly in heap?
		# Yes we could and maybe should. Would make implementation
		# less memory intensive. Maybe nice to have it for debugging?
	def __init__(self, net: Model, lambda_: float, expansions: int):
		"""Init data structure, save params

		:param net: Neural network whose value output is used as heuristic h
		:param lambda_: The weighting factor in [0,1] that weighs the cost from start node g(x)
		:param expansions: Number of expansions to perform at a time
		"""
		super().__init__(net)
		self.lambda_ = lambda_
		self.expansions = expansions

	@no_grad
	def search(self, state: np.ndarray, time_limit: float=None, max_states: int=None) -> bool:
		self.tt.tick()
		self.reset()
		self.net.eval()
		assert time_limit or max_states
		time_limit = time_limit or 1e10
		max_states = max_states or int(1e10)

		self.indices[state.tostring()] = 1
		self.states[1] = state
		if Cube.is_solved(state): return True
		# First node given cost 0: Should not matter; just to avoid np.empty weirdness
		self.G[1], self.H[1] = 0, 0
		heapq.heappush( self.open_queue, (self.cost(1), 1) )
		while self.tt.tock() < time_limit and len(self) + self.expansions <= max_states:
			self.tt.profile("Remove nodes from open priority queue")
			n_remove = min( len(self.open_queue), self.expansions )
			expand_idcs = np.array([ heapq.heappop(self.open_queue)[1] for _ in range(n_remove) ], dtype=int)

			self.open_[expand_idcs] = False
			self.closed[expand_idcs] = True
			self.tt.end_profile("Remove nodes from open priority queue")
			#TODO:
				# A decision to be made: Where do we check for solved?
				# In A* normally and in the paper it seems to be done on the expanded nodes
				# and thus not on their substates. (Thus it waits at least 1 iteration after discovering the node to terminate)
				# This surely helps finding the shortest path in some (small) problems,
				# but seems wasteful and not greedy enough on this problem.
				# Right now, I'm going with checking this already when adding the substates
				# as we don't have just as much focus on the solution length and are probably
				# more interested in running time/solve percentage.

			is_won = self.expand_batch(expand_idcs)
			if is_won: #🦀🦀🦀WE DID IT BOIS🦀🦀🦀
				i = self.indices[ Cube.get_solved().tostring() ]
				while i != 1:
					self.action_queue.appendleft(
						self.parent_actions[i]
					)
					i = self.parents[i]
				return True
		return False

	def expand_batch(self, expand_idcs: np.ndarray) -> True:
		"""
		Expands to the neighbors of each of the states in
		:param expand_idcs:
		:return: True iff. solution was found in this expansion
		(Very loose) pseudo code:
		```
		1. Calculate substates for all the batch bois
		2. Check which substates are won, seen and not seen
		3. FOR the unseen
			Set the state as their parent and set their G
			Calculate their H and add to open-list with correct cost
		4. FOR the seen
			IF substate is closed AND its's G is lower than state's G:
				Update state's G and set the substate as it's parent
			ELSE IF substate is open AND state's G is lower than substate's G
				Update substate's G and set state as it's parent
		```
		"""
		expand_states = self.states[expand_idcs]
		expand_size = len(expand_idcs)
		while len(self) + expand_size * Cube.action_dim > len(self.states):
			self.increase_stack_size()

		self.tt.profile("Calculate substates")
		parent_idcs = np.repeat(expand_idcs, Cube.action_dim, axis=0)
		substates = Cube.multi_rotate(
			self.states[parent_idcs],
			*Cube.iter_actions(expand_size)
		)
		actions_taken = np.tile(np.arange(Cube.action_dim), expand_size)
		self.tt.end_profile("Calculate substates")

		self.tt.profile("Find new substates")
		substate_strs = [s.tostring() for s in substates]
		get_substate_strs = lambda bools: [s for s, b in zip(substate_strs, bools) if b]
		seen_substates = np.array([s in self.indices for s in substate_strs])
		unseen_substates = ~seen_substates
			# Handle duplicates
		last_occurences		= np.array([s not in substate_strs[i+1:] for i, s in enumerate(substate_strs)])
		last_seen		= last_occurences & seen_substates
		last_unseen		= last_occurences & unseen_substates
		self.tt.end_profile("Find new substates")

		self.tt.profile("Add substates to data structure")
		new_states		= substates[last_unseen]
		new_states_idcs		= len(self) + np.arange(last_unseen.sum()) + 1
		new_idcs_dict		= { s: i for i, s in zip(new_states_idcs, get_substate_strs(last_unseen)) }
		self.indices.update(new_idcs_dict)
		substate_idcs		= np.array([self.indices[s] for s in substate_strs])
		old_states_idcs		= substate_idcs[last_seen]

		self.states[new_states_idcs] = substates[last_unseen]
		self.tt.end_profile("Add substates to data structure")

		self.tt.profile("Forward pass new states")
		self.H[new_states_idcs] = self.batched_H(new_states)
		self.tt.end_profile("Forward pass new states")

		self.tt.profile("Update new state values") # TODO: Test these lines: Is my numpy-fu up to the game?
		new_parent_idcs = parent_idcs[last_unseen]
		self.G[new_states_idcs] = self.G[new_parent_idcs] + 1
		self.parent_actions[new_states_idcs] = actions_taken[last_unseen]
		self.parents[new_states_idcs] = new_parent_idcs
			# Add the new states to "open" priority queue
		for i in new_states_idcs:
			heapq.heappush( self.open_queue, (self.cost(i), i) )
		self.open_[new_states_idcs] = True
		self.tt.end_profile("Update new state values")

		self.tt.profile("Check whether won") #TODO: Consider the location of this "won" check. See comment in search
		solved_substates = Cube.multi_is_solved(new_states)
		if solved_substates.any():
			return True
		self.tt.end_profile("Check whether won")

		# TODO: Test this section
		# Need to especially test the parent stuff and that I have made no weird indexing errors.
		# Also: Have I understood the algorithm cases correctly?
		self.tt.profile("Old states: Update parents and G")

		local_old_idcs = np.arange(len(substates))[last_seen] #Old idcs corresponding to last_seen
		old_parent_idcs = parent_idcs[local_old_idcs]
			# Update for the cases where the substate is a new shortcut to its parent
		shortcuts = (self.G[old_states_idcs] + 1 < self.G[old_parent_idcs]) & self.closed[old_states_idcs]
		self.G[ old_parent_idcs[shortcuts] ] = self.G[ old_states_idcs[shortcuts] ] + 1
		self.parent_actions[ old_parent_idcs[shortcuts] ] = Cube.rev_actions( actions_taken[ local_old_idcs[shortcuts] ] )
		self.parents[ old_parent_idcs[shortcuts] ] = old_states_idcs[shortcuts]

			# Update for the cases where a faster way has been found to the substate
		new_ways =  (self.G[old_parent_idcs] + 1 < self.G[old_states_idcs] ) & self.open_[old_states_idcs]
		self.G[ old_states_idcs[new_ways] ] = self.G[ old_parent_idcs[new_ways] ] + 1
		self.parent_actions[ old_states_idcs[new_ways] ] = actions_taken[ local_old_idcs[new_ways] ]
		self.parents[ old_states_idcs[new_ways] ] = old_parent_idcs[new_ways]

		self.tt.end_profile("Old states: Update parents and G")

		return False

	def cost(self, i: np.ndarray):
		"""The A star costs of the states saved in the indeces corresponding to the vector i"""
		return self.H[i] + self.lambda_ * self.G[i]

	@no_grad
	def batched_H(self, states: np.ndarray):
		"""Heuristics calculater.
		Uses the value neural network. -value is regarded as the distance heuristic
		:param states: (batch size, *(cube_dimensions)) of states
		"""
		states = Cube.as_oh(states)
		J = -self.net(states, value=True, policy=False)
		return J.cpu().squeeze().detach().numpy()

	def reset(self):
		super().reset()
		self.open_queue = list()
		self.indices   = dict()

		self.states    = np.empty((self._stack_expand, *Cube.shape()), dtype=Cube.dtype)
		self.parents = np.empty(self._stack_expand, dtype=int)
		self.parent_actions = np.zeros(self._stack_expand, dtype=int)
		self.G         = np.empty(self._stack_expand)
		self.H         = np.empty(self._stack_expand)
		self.closed    = np.zeros(self._stack_expand, dtype=bool)
		self.open_     = np.zeros(self._stack_expand, dtype=bool)

	def increase_stack_size(self):
		expand_size    = len(self.states)

		self.states	   = np.concatenate([self.states, np.empty((expand_size, *Cube.shape()), dtype=Cube.dtype)])
		self.parents   = np.concatenate([self.parents, np.zeros(expand_size, dtype=int)])
		self.parent_actions   = np.concatenate([self.parent_actions, np.zeros(expand_size, dtype=int)])
		self.G         = np.concatenate([self.G, np.empty(expand_size)])
		self.H         = np.concatenate([self.H, np.empty(expand_size)])
		self.closed    = np.concatenate([self.closed, np.zeros(expand_size, dtype=bool)])
		self.open_     = np.concatenate([self.open_, np.zeros(expand_size, dtype=bool)])

	@classmethod
	def from_saved(cls, loc: str, use_best: bool, lambda_: float, expansions: int):
		net = Model.load(loc, load_best=use_best).to(gpu)
		return cls(net, lambda_=lambda_, expansions=expansions)

	def __len__(self): return len(self.indices)

	def __str__(self): return f'AStar(lambda={self.lambda_}, N={self.expansions})'

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

		while self.tt.tock() < time_limit and len(self) + self.workers * self.depth <= max_states:
			# Expand from current best state
			paths, states, states_oh, solved = self.expand(state)
			# Break if solution is found
			if solved != (-1, -1):
				self.action_queue += deque(paths[solved[0]][:solved[1]])
				return True
			# Update state with the high ground
			v = self.net(states_oh, policy=False).cpu().squeeze()
			best_value_index = int(v.argmax())
			state = states[best_value_index]
			worker, depth = best_value_index // self.depth, best_value_index % self.depth
			self.action_queue += deque(paths[worker][:depth])

		return False

	def _get_indices(self, depth: int) -> np.ndarray:
		return np.arange(self.workers) * self.depth + depth

	def expand(self, state: np.ndarray) -> (list, np.ndarray, torch.tensor, tuple):
		# Initialize needed data structures
		states = Cube.repeat_state(state, self.workers)
		states_oh = Cube.as_oh(states)
		paths = [[] for _ in range(self.workers)]
		new_states = np.empty((self.workers * self.depth, *Cube.shape()), dtype=Cube.dtype)
		new_states_oh = torch.empty(self.workers * self.depth, Cube.get_oh_shape(), dtype=torch.float, device=gpu)
		# Expand for self.depth iterations
		for d in range(self.depth):
			# Use epsilon-greedy to decide where to use policy and random actions
			use_random = np.random.choice(2, self.workers, p=[1-self.epsilon, self.epsilon]).astype(bool)
			use_policy = ~use_random
			actions = np.empty(self.workers, dtype=int)
			# Random actions
			actions[use_random] = np.random.randint(0, Cube.action_dim, use_random.sum())
			# Policy actions
			p = self.net(states_oh[use_policy], value=False).cpu().numpy()
			actions[use_policy] = p.argmax(axis=1)
			# Update paths
			[path.append(a) for path, a in zip(paths, actions)]

			# Expand using selected actions
			faces, dirs = Cube.indices_to_actions(actions)
			states = Cube.multi_rotate(states, faces, dirs)
			states_oh = Cube.as_oh(states)
			solved_states = Cube.multi_is_solved(states)
			if np.any(solved_states):
				self._explored_states += (d+1) * self.workers
				w = np.where(solved_states)[0][0]
				return paths, None, None, (w, d+1)
			new_states[self._get_indices(d)] = states
			new_states_oh[self._get_indices(d)] = states_oh
		self._explored_states += len(new_states)

		return paths, new_states, new_states_oh, (-1, -1)

	@classmethod
	def from_saved(cls, loc: str, use_best: bool, epsilon: float, workers: int, depth: int):
		net = Model.load(loc, load_best=use_best).to(gpu)
		return cls(net, epsilon=epsilon, workers=workers, depth=depth)

	def __str__(self):
		return f"EGVM (e={self.epsilon}, w={self.workers}, d={self.depth})"

