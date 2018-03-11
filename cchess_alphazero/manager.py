import argparse

from logging import getLogger

from cchess_alphazero.lib.logger import setup_logger
from cchess_alphazero.config import Config

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'play']


