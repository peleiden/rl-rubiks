import os
from shutil import rmtree
from librubiks.utils import seedsetter

class MainTest:
	@classmethod
	def setup_class(cls):
		os.makedirs("local_tests", exist_ok = True)
		seedsetter()
		repo_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )

		if 'PYTHONPATH' not in os.environ: os.environ['PYTHONPATH'] = ''
		if repo_path not in os.environ['PYTHONPATH']: os.environ['PYTHONPATH'] += f':{repo_path}'
	@classmethod
	def teardown_class(cls):
		rmtree('local_tests', onerror=cls.ignore_absentee)

	@staticmethod
	def ignore_absentee(func, path, exc_inf):
		except_instance = exc_inf[1]
		if isinstance(except_instance, FileNotFoundError): return
		raise except_instance
