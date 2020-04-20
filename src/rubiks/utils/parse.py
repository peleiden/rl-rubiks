import os, sys
from argparse import ArgumentParser, RawTextHelpFormatter
from configparser import ConfigParser

from pprint import pformat


class Parser:
	"""
	Maintains multiple parsers to allow a workflow of mixing use of config files and CLI arguments.
	Receives a dict of options. This should be a dict of dicts where each dict corresponds to an option with the key as the name.
	In the dict for each option, the key 'default' should store the default value and all other keys should correspond to
	kwargs in ArgumentParser.add_argument.

	Can read from single .ini file and CLI arguments. CLI arguments overwrite settings in all experiments defined if ini file.
	In .ini file, defaults for all runs can be set in [DEFAULT] section and multiple runs can be added with their own section.

	The parser returns a list of dicts of the parsed settings for the experiments.


	Quick example:
	A file `main.py` could contain:
	```
	options = {
	location: {default: 'local_train', help: 'save_location', type: str},
	learning_rate: {default: 1.5e-3, help: 'Controls size of parameter update', type: float}
	}
	parser = Parser(options)
	experiments = parser.parse()

	```
	This could then by run by
	`python main.py --learning_rate 1e-5`
	or by
	`python main.py --config cfg.ini`
	where `cfg.ini` could contain

	```
	[DEFAULT]
	location = data/my_big_experiment
	[RUN1]
	learning_rate = 1e-4
	[RUN2]
	learning_rate = 1e-5
	```

	"""
	def __init__(self, options: dict, name: str = "Experiment",  description: str = "Run experiments with following options", show_defaults: bool = True):
		self.options = options
		self.defaults = dict()
		self.save_location = ''
		self.name = name

		#Seperate parser for only receiving the config file
		self.config_receiver = ArgumentParser(add_help = False)
		self.config_receiver.add_argument('--config', help="Location of configuration file to use (if any). Config file should follow .ini format.", metavar='FILE')

		#Main parser for CLI arguments
		self.argparser = ArgumentParser(description = description, formatter_class = RawTextHelpFormatter, parents = [self.config_receiver])

		for argname, settings in self.options.items():
			self.defaults[argname] = settings.pop('default')

			if 'help' in settings and show_defaults:
				settings['help'] += f"\n  Default='{self.defaults[argname]}'"

			self.argparser.add_argument(f'--{argname}', **settings)

		self.configparser = ConfigParser()

	def parse(self, document = True):
		conf_arg, args = self.config_receiver.parse_known_args()

		experiments, with_config = self._read_config(conf_arg, args)

		location = ''
		if not experiments: #If configparser set nothing or only set defaults
			self.argparser.set_defaults(**self.defaults)
			args = self.argparser.parse_args(args)
			if args.location:
				self.defaults['location'] = args.location
				location = args.location
			del args.config
			experiments.append({'name': self.name, **vars(args)})


		if document: self._document_settings(with_config, location)

		return experiments

	def _read_config(self, conf_arg, args):
		experiments = list()
		with_config = False

		if conf_arg.config:
			with_config = True
			if not self.configparser.read([conf_arg.config]):
				raise FileNotFoundError(f"Could not find config file {conf_arg.config}")

			# User set DEFAULT section should overwrite the defaults
			self.defaults = {**self.defaults, **dict(self.configparser.items("DEFAULT"))}

			# Each other section corresponds to an experiment
			for experiment_name in self.configparser.sections():
				options = {**self.defaults, **dict(self.configparser.items(experiment_name))}
				self.argparser.set_defaults(**options)
				exp_args = self.argparser.parse_args(args)

				#For multiple experiments, subfolders are nice
				if exp_args.location: exp_args.location = f"{exp_args.location}/{experiment_name.lower()}"

				del exp_args.config
				experiments.append({'name': experiment_name, **vars(exp_args)})

		return experiments, with_config


	def _document_settings(self, with_config: bool, location: str):
		"""
		Saves all settings used for experiments for reproducability.
		"""
		if location: self.save_location = location #To allow both for forced location and the automatic, default location
		elif 'location' in self.defaults: self.save_location = self.defaults['location']

		os.makedirs(self.save_location, exist_ok = True)

		with open(f"{self.save_location}/{self.name}_config.ini", 'w') as f:
			if with_config: self.configparser.write(f)
			f.write(f"\n# Run command\n # {' '.join(sys.argv)}\n")
			str_defaults = pformat(self.defaults).replace('\n', '\n# ')
			f.write(f"\n# Default configuration values at run\n # {str_defaults}")


