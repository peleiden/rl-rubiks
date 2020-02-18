import torch

from ..rubiks.example import ExampleAdder
from ..rubiks.utils.ticktock import TickTock



class TestExampleAdder:
	def test_init(self):
		fill = 81213123
		expected = torch.ones(3,3)*fill
		ex = ExampleAdder(fill)

		#Testing that initializaiton is complete 
		assert torch.equal(expected, ex.tensor)

	def test_add(self):
		fill = 9
		expected = torch.ones(3,3)*fill+3
		ex = ExampleAdder(fill)
		
		#Testing that standard value is 3
		assert torch.equal(expected,  ex.add_to_tensor())

class TestTickTock:
	def test_tick_tock(self):
		tt = TickTock()
		tt.tick()
		torch.zeros(100)
		tt.tock(True)

if __name__ == "__main__":
	TestTickTock().test_tick_tock()
