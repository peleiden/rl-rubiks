from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

from ast import literal_eval
import numpy as np

from src.rubiks.cube.cube import Cube

app = Flask(__name__)
api = Api(app)
CORS(app)

def get_state_dict(state: np.ndarray or list):
	if type(state) == list:
		state = np.array(state)
	return jsonify({
		"state": Cube.as633(state).tolist(),
		"state20": state.tolist(),
	})

class CubeService(Resource):
	def get(self):
		return get_state_dict(Cube.get_solved())
	def post(self):
		data = literal_eval(request.data.decode("utf-8"))
		action = data["action"]
		state = data["state20"]
		new_state = Cube.rotate(state, *Cube.action_space[action])
		return get_state_dict(new_state)

api.add_resource(CubeService, "/cube")

if __name__ == "__main__":
	app.run()


