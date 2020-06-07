import os
from ast import literal_eval
from wget import download

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS

import numpy as np
import torch

from librubiks import cube
from librubiks.solving.agents import RandomSearch, BFS, PolicySearch, MCTS, AStar, EGVM, ValueSearch

app = Flask(__name__)
api = Api(app)
CORS(app)

net_loc = "local_net"
os.makedirs(net_loc, exist_ok=True)
url = "https://github.com/peleiden/rubiks-models/blob/master/fcnew/%s?raw=true"
download(url % "model.pt", net_loc)
download(url % "config.json", net_loc)

agents = [
	{ "name": "AStar", "agent": AStar.from_saved(net_loc, use_best=False, lambda_=0.2, expansions=50) },
	{ "name": "MCTS", "agent": MCTS.from_saved(net_loc, use_best=False, c=0.6, search_graph=True) },
	{ "name": "Greedy policy", "agent": PolicySearch.from_saved(net_loc, use_best=False) },
	{ "name": "Greedy value", "agent": ValueSearch.from_saved(net_loc, use_best=False) },
	{ "name": "EGVM", "agent": EGVM.from_saved(net_loc, use_best=False, epsilon=0.5, workers=10, depth=100) },
	{ "name": "BFS", "agent": BFS() },
	{ "name": "Random actions", "agent": RandomSearch()},
]

@app.route("/")
def index():
	return "<a href='https://peleiden.github.io/rl-rubiks' style='margin: 20px'>Go to main page</a>"

@app.route("/info")
def get_info():
	return jsonify({
		"cuda": torch.cuda.is_available(),
		"agents": [x["name"] for x in agents],
	})

@app.route("/solve", methods=["POST"])
def solve():
	data = literal_eval(request.data.decode("utf-8"))
	time_limit = data["timeLimit"]
	agent = agents[data["agentIdx"]]["agent"]
	state = np.array(data["state"], dtype=cube.dtype)
	solution_found = agent.search(state, time_limit)
	return jsonify({
		"solution": solution_found,
		"actions": [int(x) for x in agent.action_queue],
		"exploredStates": len(agent),
	})


if __name__ == "__main__":
	app.run(port=8000, debug=False)
