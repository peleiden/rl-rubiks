import os, sys
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

from ast import literal_eval
import numpy as np
import torch

from src.rubiks.solving.agents import Agent, DeepAgent
from src.rubiks.solving.search import RandomDFS, BFS, PolicySearch, MCTS
from src.rubiks.cube.cube import Cube

app = Flask(__name__)
api = Api(app)
CORS(app)

net_loc = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Get parent folder
	'data',
	'hpc-20-04-12'
)
agents = [
	{ "name": "DeepCube", "agent": DeepAgent(MCTS.from_saved(net_loc, 0.6, 0.005, True, 10)) },
	{ "name": "Greedy policy", "agent": DeepAgent(PolicySearch.from_saved(net_loc, False)) },
	{ "name": "BFS", "agent": Agent(BFS()) },
	{ "name": "Random actions", "agent": Agent(RandomDFS()) },
	{ "name": "Stochastic policy", "agent": DeepAgent(PolicySearch.from_saved(net_loc, True)) },
]

def as69(state: np.ndarray):
	# Nice
	return Cube.as633(state).reshape((6, 9))


def get_state_dict(state: np.ndarray or list):
	state = np.array(state)
	return jsonify({
		"state": as69(state).tolist(),
		"state20": state.tolist(),
	})

@app.route("/")
def index():
	return "<a href='https://asgerius.github.io/rl-rubiks' style='margin: 20px'>Gå til hovedside</a>"

@app.route("/info")
def get_info():
	return jsonify({
		"cuda": torch.cuda.is_available(),
		"agents": [x["name"] for x in agents],
	})

@app.route("/solved")
def get_solved():
	return get_state_dict(Cube.get_solved())

@app.route("/action", methods=["POST"])
def act():
	data = literal_eval(request.data.decode("utf-8"))
	action = data["action"]
	state = data["state20"]
	new_state = Cube.rotate(state, *Cube.action_space[action])
	return get_state_dict(new_state)

@app.route("/scramble", methods=["POST"])
def scramble():
	data = literal_eval(request.data.decode("utf-8"))
	depth = data["depth"]
	state = np.array(data["state20"])
	states = []
	for _ in range(depth):
		action = np.random.randint(Cube.action_dim)
		state = Cube.rotate(state, *Cube.action_space[action])
		states.append(state)
	finalOh = states[-1]
	states = np.array([as69(state) for state in states])
	return jsonify({
		"states": states.tolist(),
		"finalState20": finalOh.tolist(),
	})

@app.route("/solve", methods=["POST"])
def solve():
	data = literal_eval(request.data.decode("utf-8"))
	time_limit = data["timeLimit"]
	agent = agents[data["agentIdx"]]["agent"]
	state = np.array(data["state20"])
	solution_found, n_steps = agent.generate_action_queue(state, time_limit)
	actions = list(agent.actions())
	states = []
	if solution_found:
		for action in actions:
			state = Cube.rotate(state, *action)
			states.append(state)
	finalOh = states[-1] if states else state
	states = np.array([as69(state) for state in states])
	return jsonify({
		"solution": solution_found,
		"actions": actions,
		"searchedStates": len(agent),
		"states": states.tolist(),
		"finalState20": finalOh.tolist(),
	})


if __name__ == "__main__":
	app.run(debug=True)
