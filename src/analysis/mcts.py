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
	state, _, _ = Cube.scramble(50)
	net = Model(ModelConfig()).to(gpu)
	searcher = MCTS(net)
	searcher.search(state, 1)
	print(searcher.tt)

if __name__ == "__main__":
	seedsetter()
	analyse_mcts()
	



