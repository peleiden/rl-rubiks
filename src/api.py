import os, sys
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

from ast import literal_eval
import numpy as np
import torch

from src.rubiks.solving.agents import RandomAgent, SimpleBFS, DeepCube, PolicyCube
from src.rubiks.cube.cube import Cube

app = Flask(__name__)
api = Api(app)
CORS(app)

net_loc = os.path.join(sys.path[0], "rubiks", "local_train")
tree_agents = {
	"Random": RandomAgent,
	"BFS": SimpleBFS,
	"DeepCube": DeepCube
}

def as69(state: np.ndarray):
	# Nice
	return Cube.as633(state).reshape((6, 9))


def get_state_dict(state: np.ndarray or list):
	state = np.array(state)
	return jsonify({
		"state": as69(state).tolist(),
		"state20": state.tolist(),
	})

@app.route("/info")
def get_info():
	return jsonify({
		"cuda": torch.cuda.is_available(),
		"treeAgents": ["Random", "BFS", "DeepCube"],
		"stateAgents": ["PolicyCube"],
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
		action = np.random.randint(12)
		state = Cube.rotate(state, *Cube.action_space[action])
		states.append(state)
	finalOh = states[-1]
	states = np.array([as69(state) for state in states])
	return jsonify({
		"states": states.tolist(),
		"finalState20": finalOh.tolist(),
	})



if __name__ == "__main__":
	app.run()


