from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

from src.rubiks.cube.cube import Cube

app = Flask(__name__)
api = Api(app)
CORS(app)

class CubeService(Resource):
	def get(self):
		return Cube.get_solved().tolist()
	def post(self):
		print(request.form)
		return Cube.get_solved().tolist()

api.add_resource(CubeService, "/cube")

if __name__ == "__main__":
	app.run()


