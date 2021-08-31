import os
from itertools import chain

from config.default import config as default_config
from config.development import config as dev_config
from config.production import config as prod_config
from config.env import config as envs

environment = os.environ.get('PYTHON_ENV', 'development')
environment_config = None
if environment == 'development':
    environment_config = dev_config
if environment == 'production':
    environment_config = prod_config

config = dict(chain(default_config.items(), environment_config.items(), envs.items()))