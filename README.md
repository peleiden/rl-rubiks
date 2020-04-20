# Solving Rubik's cube using Deep Reinforcement Learning

###### Second year project at the bachelor in Artificial Intelligence and Data, Technical University of Denmark.

## Demo
Visit the [Demo page](https://asgerius.github.io/rl-rubiks/) to see the trained Reinforcement Learning agents in action.

## Reproducing the results 

### Setup
Clone the repo:
```
git clone https://github.com/asgerius/rl-rubiks
```
For the modules to work, you *might* need to add the main repo folder to your PYTHONPATH. 
In \*nix-systems, you can add the following line to your shell login script:
```
export PYTHONPATH=$PYTHONPATH:<path where you cloned>/rl-rubiks
```
If your shell is bash, this can be achieved by running
```
echo "export PYTHONPATH=\$PYTHONPATH:<path where you cloned>/rl-rubiks" >> ~/.bashrc
```
On Windows, you can follow some of the instructions in [this SO thread](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages) 

### Starting training
The training can be started using `src/rubiks/runtrain.py`. From the main repo folder, run
```
python src/rubiks/runtrain.py --help
```
for help on options. In many cases, using the parameters from a configuration file is preferable. An example of using this to run two trainings, can be seen be running
```
python src/rubiks/runtrain.py --config data/configs/train_ex.ini
```
### Evaluation of agents

To evaluate the performance of one or more agents, a solving experiment can be started using `src/rubiks/runeval.py`. From the main repo folder, run
```
python src/rubiks/runeval.py --help
```
for help on options. This also works with configuration files; an example of which can be seen be running
```
python src/rubiks/runeval.py --config data/configs/eval_ex.ini
```
