"""Logging configuration"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


_LOG_HANDLER = None
_STDOUT_WRAPPER = None
_STDERR_WRAPPER = None

def get_logger(name=__name__, log_level="DEBUG"):
    global _STDOUT_WRAPPER
    global _STDERR_WRAPPER

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper()))

    handler = _get_log_handler()

    # Avoid duplicate handlers.
    logger.handlers = [handler]

    # Redirect stdout/stderr once.
    if _STDOUT_WRAPPER is None:
        _STDOUT_WRAPPER = StreamToLogFile(handler)
        _STDERR_WRAPPER = StreamToLogFile(handler)

        sys.stdout = _STDOUT_WRAPPER
        sys.stderr = _STDERR_WRAPPER

    return logger


class StreamToLogFile:
    """Write stdout/stderr directly to the rotating log file."""

    def __init__(self, handler):
        self.handler = handler

    def write(self, message):
        if not message:
            return

        # Click and some other libraries may write bytes rather than str.
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")

        self.handler.acquire()
        try:
            # Rotate if necessary.
            record = logging.LogRecord(
                name="stream",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None,
            )

            if self.handler.shouldRollover(record):
                self.handler.doRollover()

            self.handler.stream.write(message)
            self.handler.stream.flush()

        finally:
            self.handler.release()

    def flush(self):
        self.handler.acquire()
        try:
            if self.handler.stream is not None:
                self.handler.stream.flush()
        finally:
            self.handler.release()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return self.handler.encoding

    def __getattr__(self, name):
        # Let libraries such as Click access other normal stream
        # attributes/methods if necessary.
        return getattr(sys.__stdout__, name)


def _get_log_handler():
    global _LOG_HANDLER

    if _LOG_HANDLER is None:
        log_file = os.path.join(
            os.path.dirname(__file__),
            "saxswaxs-server",
        )

        _LOG_HANDLER = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )

        _LOG_HANDLER.suffix = "%Y-%m-%d.log"

        _LOG_HANDLER.setFormatter(
            logging.Formatter(
                "{asctime}: {name:20} (L{lineno}): "
                "{levelname}: {message}",
                datefmt="%Y-%m-%d %H:%M:%S",
                style="{",
            )
        )

    return _LOG_HANDLER
