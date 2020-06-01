# Frontend

<a href="https://peleiden.github.io/rl-rubiks">Demo page</a>

The frontend has two purposes: Demo our results and show things not well suited for a report, such as the value target GIF.

## Setup a local backend

By default a Python backend hosted on <a href="https://heroku.com">Heroku</a> is used, but this has access two only two fairly slow CPU cores. The lack of CUDA support is especially limiting for the agents using a neural network, who will need much more time two solve deeply scrambled cubes. If you want to use the demo with CUDA support, you can start your own local server. From the main repo folder, run
```
pip install -r rlrequirements.txt  # Install dependencies for the librubiks module
pip install wget flask flask_cors flask_restful  # Install dependencies for running the server
python librubiks/api.py
```
It is possible to connect to a local server from the demo page, so you do not need to run the frontend locally to use a local backend.

## Run the frontend locally

In case you want to start the frontend locally, you can do so by running
```
cd frontend
npm install
npm start
```
from the main repo folder and opening your browser at `localhost:4200`.
