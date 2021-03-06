import logging
from environs import Env

env = Env()
env.read_env()
ROOT = env.str("ROOT_PATH")


class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.__level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.__level


def get_logger(handle_info=True, handle_errors=True) -> logging.Logger:
    logger = logging.getLogger(__name__)
    format_str = u'%(filename)s [LINE:%(lineno)d] #%(levelname)-8s [%(asctime)s]  %(message)s'
    formatter = logging.Formatter(format_str)
    logger.setLevel(level=logging.INFO)
    if handle_info:
        file_info_handler = logging.FileHandler(f'{ROOT}/info.log')
        file_info_handler.setLevel(logging.INFO)
        file_info_handler.setFormatter(formatter)
        file_info_handler.addFilter(LevelFilter(logging.INFO))
        logger.addHandler(file_info_handler)
    if handle_errors:
        file_error_handler = logging.FileHandler(f'{ROOT}/errors.log')
        file_error_handler.setLevel(logging.ERROR)
        file_error_handler.setFormatter(formatter)
        logger.addHandler(file_error_handler)
    logging.basicConfig(format=format_str, level=logging.INFO)
    return logger
