# Hacky import code needed to import from sibling dir
import sys, os
sys.path.append(os.path.join(sys.path[0], '..', 'src'))


import torch 

from example import ExampleAdder



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