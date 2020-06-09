import torch
from tests import MainTest
from librubiks import no_grad
from librubiks.cube import get_is2024, set_is2024, store_repr, restore_repr, with_used_repr


class TestRubiks(MainTest):
	is2024: bool

	def test_repr(self):
		ini_repr = get_is2024()
		store_repr()
		assert get_is2024() == ini_repr
		set_is2024(not ini_repr)
		assert get_is2024() != ini_repr
		restore_repr()
		assert get_is2024() == ini_repr

		set_is2024(True)
		self.is2024 = True
		assert self._get_is2024()
		self.is2024 = False
		assert not self._get_is2024()
		assert get_is2024()

	@with_used_repr
	def _get_is2024(self):
		return get_is2024()

	def test_no_grad(self):
		torch.set_grad_enabled(True)
		assert torch.is_grad_enabled()
		assert not self._get_grad()
		assert torch.is_grad_enabled()

	@no_grad
	def _get_grad(self):
		return torch.is_grad_enabled()
