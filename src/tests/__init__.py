import os
from shutil import rmtree
from src.rubiks.utils import seedsetter

class MainTest:
	@classmethod
	def setup_class(cls):
		os.makedirs("local_tests", exist_ok = True)
		seedsetter()

	@classmethod
	def teardown_class(cls):
		rmtree('local_tests', onerror=cls.ignore_absentee)

	@staticmethod
	def ignore_absentee(func, path, exc_inf):
		except_instance = exc_inf[1]
		if isinstance(except_instance, FileNotFoundError): return
		raise except_instance
