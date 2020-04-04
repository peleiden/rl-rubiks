from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

from src.rubiks.cube.cube import Cube

app = Flask(__name__)
api = Api(app)
CORS(app)

class CubeService(Resource):
	state = Cube.get_solved()
	def get(self):
		self.state = Cube.get_solved()
		return Cube.as633(self.state).tolist()
	def post(self):
		action = int(request.data)
		self.state = Cube.rotate(self.state, *Cube.action_space[action])
		return Cube.as633(self.state).tolist()

api.add_resource(CubeService, "/cube")

if __name__ == "__main__":
	app.run()


