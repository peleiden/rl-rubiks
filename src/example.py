import torch 



class ExampleAdder:
	def __init__(self, filler_num: int):
		self.tensor = torch.ones(3,3) * filler_num

	def add_to_tensor(self, addend: int = 3):
		return self.tensor + addend

if __name__ == "__main__":
	ex = ExampleAdder(5)
	print(
		ex.add_to_tensor(2)
	)