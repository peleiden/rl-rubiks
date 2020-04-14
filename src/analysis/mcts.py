from src.rubiks import gpu
from src.rubiks.cube.cube import Cube
from src.rubiks.model import Model, ModelConfig
from src.rubiks.solving.search import Searcher, MCTS
from src.rubiks.utils import seedsetter
from src.rubiks.utils.logger import Logger
from src.rubiks.utils.ticktock import TickTock

tt = TickTock()
log = Logger("data/local_analyses/mcts.log", "Analyzing MCTS")

def analyse_mcts():
	time_limit = 1
	state, _, _ = Cube.scramble(50)
	# net = Model(ModelConfig()).to(gpu).eval()
	net = Model.load("data/hpc-20-04-12").to(gpu).eval()
	searcher = MCTS(net)
	searcher.search(state, time_limit)
	log(searcher.tt)
	log(f"Tree size after {time_limit} s: {len(searcher.states)}")

if __name__ == "__main__":
	seedsetter()
	analyse_mcts()
	



