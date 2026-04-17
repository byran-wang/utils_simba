
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
import logging
import os


def fetch_logger(logger_name="", logger_file=""):
    # Create a custom logger
    if logger_name != "":
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s')
    console_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(console_handler)    


    if logger_file != "":
            # Create a file handler
        file_handler = logging.FileHandler(logger_file)
        file_handler.setLevel(logging.DEBUG)    
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """A command line formatter with different colors for each level."""

    def __init__(self):
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",  # cyan,
            logging.INFO: f"{reset}\033[32m",  # green
            logging.WARNING: f"{reset}\033[33m",  # yellow
            logging.ERROR: f"{reset}\033[31m",  # red
            logging.CRITICAL: f"{reset}\033[35m",  # magenta
        }
        fmt_str = "{color}%(levelname)s %(asctime)s %(filename)s:%(lineno)d :%(funcName)s {reset} %(message)s"
        self.formatters = {
            level: logging.Formatter(fmt_str.format(color=color, reset=reset), datefmt="%H:%M:%S")
            for level, color in colors.items()
        }
        self.default_formatter = self.formatters[logging.INFO]

    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)

def PlainFormatter():
    return logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s:%(lineno)d :%(funcName)s %(message)s",
        datefmt="%H:%M:%S",
    )

def get_logger(name, level=logging.INFO):
    """A command line logger."""
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        assert (
            level in LOG_LEVELS
        ), f"Invalid LOG_LEVEL: {level}, must be one of {list(LOG_LEVELS.keys())}"
        level = LOG_LEVELS[level]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(PlainFormatter())
    logger.addHandler(ch)
    return logger
