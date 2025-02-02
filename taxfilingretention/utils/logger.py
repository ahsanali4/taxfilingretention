import logging
import sys
from logging import FileHandler

LOG_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(module)s — %(levelname)s — %(message)s"
)
USER_LOG_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(module)s — %(levelname)s — %(message)s"
)
CONSOLE_FORMATTER = logging.Formatter("%(asctime)s|%(name)s|%(module)s|%(levelname)s|%(message)s")
LOG_FILE = "service.log"


class LevelFilter(object):
    """Filter the desire logs from logging"""

    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno != self.level


class TracebackInfoFilter(logging.Filter):
    """Clear or restore the exception on log records"""

    def __init__(self, clear=True):
        self.clear = clear

    def filter(self, record):
        if self.clear:
            record._exc_info_hidden, record.exc_info = record.exc_info, None
            # clear the exception traceback text cache, if created.
            record.exc_text = None
        elif hasattr(record, "_exc_info_hidden"):
            record.exc_info = record._exc_info_hidden  # type: ignore
            del record._exc_info_hidden  # type: ignore
        return True


class Singleton(object):
    """
    Singleton interface:
    http://www.python.org/download/releases/2.2.3/descrintro/#__new__
    """

    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


class LogManager(Singleton):

    """
    Creating a Singleton LogManager class which can be accesed in every class
    to log different levels of logs
    """

    def init(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self._get_console_handler())
        self.logger.addHandler(self._get_file_handler())
        return

    def _get_console_handler(self):
        """
        add console handler to the logging module
        set the formatter of the handler and logging level
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CONSOLE_FORMATTER)
        console_handler.setLevel(logging.DEBUG)
        return console_handler

    def _get_file_handler(self):
        """
        add file ahndler to the logging module
        set the formatter of the handler and logging level
        """
        file_handler = FileHandler(LOG_FILE)
        file_handler.setFormatter(LOG_FILE_FORMATTER)
        file_handler.setLevel(logging.INFO)
        return file_handler

    def get_logger(self):
        return self.logger


# a simple usecase

if __name__ == "__main__":
    logger = LogManager("Testing").get_logger()
    logger.info("Hello, Logger")
    logger.debug("bug occured")

    logger = LogManager("Tester").get_logger()
    logger.info("Bye, Logger")
    logger.debug("bug Cleared")
