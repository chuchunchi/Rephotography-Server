import logging
import sys
from . import utils

logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
