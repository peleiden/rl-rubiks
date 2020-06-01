# Solving Rubik's cube using Deep Reinforcement Learning

###### Second year project for Artificial Intelligence and Data, Technical University of Denmark.
![pytest_master](https://github.com/peleiden/rl-rubiks/workflows/pytest_master/badge.svg?branch=master)

## Demo
Visit the [Demo page](https://peleiden.github.io/rl-rubiks/) to see the trained Reinforcement Learning agents in action.

## Reproducing the results 

### Setup
1) Clone the repo:
```
git clone https://github.com/peleiden/rl-rubiks
cd rl-rubiks
```

2) Install requirements (use `pip3` if `pip` is not bound to your Python 3 binary):
```
pip install -r rlrequirements.txt
```
There are also some optional dependencies: `GitPython` for saving commits in logs and `networkx` and `imageio` for visualizing values of initial states over time when running with `--analysis True`.

3) (Only needed if you get `ImportError`) For the modules to work, you *might* need to add the main repo folder to your `PYTHONPATH`. 
In \*nix-systems, you can add the following line to your shell login script:
```
export PYTHONPATH=$PYTHONPATH:<path where you cloned>/rl-rubiks
```
If your shell is bash, this can be achieved by running
```
echo "export PYTHONPATH=\$PYTHONPATH:<path where you cloned>/rl-rubiks" >> ~/.bashrc
```
On Windows, you can follow some of the instructions in [this Stack Overflow thread](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages).

### Starting training
The training can be started using `runtrain.py`. From the main repo folder, run
```
python runtrain.py --help
```
to show the possible options. In many cases, using the parameters from a configuration file is preferable. We have included an example configuration file, which starts two trainings with different learning rates. Run the following command to start the trainings:
```
python runtrain.py --config configs/train_ex.ini
```

### Evaluation of agents

To evaluate the performance of one or more agents, a solving experiment can be started using `runeval.py`. From the main repo folder, run
```
python runeval.py --help
```
to show options. This also works with configuration files; an example of which can be seen by running
```
python runeval.py --config configs/eval_ex.ini
```

## Experiments
To limit the size of this repository, we have kept no models on it. Our trained models are available on the [model repository](https://github.com/peleiden/rubiks-models).
