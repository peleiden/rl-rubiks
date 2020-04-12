import torch
from src.tests.main import MainTest
from src.rubiks import cpu, gpu, get_repr, set_repr, store_repr, restore_repr, no_grad


class TestRubiks(MainTest):

	def test_repr(self):
		ini_repr = get_repr()
		store_repr()
		assert get_repr() == ini_repr
		set_repr(not ini_repr)
		assert get_repr() != ini_repr
		restore_repr()
		assert get_repr() == ini_repr

	def test_no_grad(self):
		torch.set_grad_enabled(True)
		assert torch.is_grad_enabled()
		assert not self._get_grad()
		assert torch.is_grad_enabled()

	@no_grad
	def _get_grad(self):
		return torch.is_grad_enabled()
