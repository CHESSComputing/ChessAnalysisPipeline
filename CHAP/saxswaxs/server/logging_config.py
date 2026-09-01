"""Logging configuration"""

import contextlib
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


_LOG_HANDLER = None
_STDOUT_WRAPPER = None
_STDERR_WRAPPER = None
_TASK_HANDLER = None

def get_logger(name=__name__, log_level="DEBUG"):
    global _STDOUT_WRAPPER
    global _STDERR_WRAPPER

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper()))

    handler = _get_log_handler()

    handlers = [handler]
    if _TASK_HANDLER is not None:
        handlers.append(_TASK_HANDLER)
    logger.handlers = handlers

    # Redirect stdout/stderr once.
    if _STDOUT_WRAPPER is None:
        _STDOUT_WRAPPER = StreamToLogFile(handler)
        _STDERR_WRAPPER = StreamToLogFile(handler)

        sys.stdout = _STDOUT_WRAPPER
        sys.stderr = _STDERR_WRAPPER

    return logger


@contextlib.contextmanager
def task_log_context(log_path):
    """Context manager that tees log output to a per-task append-only file.

    While active, all loggers created via :func:`get_logger` write to
    *log_path* in addition to the shared rotating log file, using the same
    formatter.  Raw stdout/stderr writes are also teed to the same file
    without any additional formatting.  The file is always opened in append
    mode so successive calls accumulate rather than overwrite.
    """
    global _TASK_HANDLER

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    task_handler = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    task_handler.setFormatter(_get_log_handler().formatter)

    prev_task_handler = _TASK_HANDLER
    _TASK_HANDLER = task_handler

    # Add to all loggers that were created by get_logger (propagate=False).
    pre_existing = [
        lgr for lgr in logging.Logger.manager.loggerDict.values()
        if isinstance(lgr, logging.Logger) and not lgr.propagate
    ]
    for lgr in pre_existing:
        lgr.addHandler(task_handler)

    try:
        yield
    finally:
        _TASK_HANDLER = prev_task_handler

        # Remove from every logger that received this handler (pre-existing
        # and any created inside the context via get_logger).
        for lgr in logging.Logger.manager.loggerDict.values():
            if isinstance(lgr, logging.Logger) and task_handler in lgr.handlers:
                lgr.handlers = [h for h in lgr.handlers if h is not task_handler]

        task_handler.close()


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

        # Tee raw output to the active task log file.
        if _TASK_HANDLER is not None:
            _TASK_HANDLER.acquire()
            try:
                _TASK_HANDLER.stream.write(message)
                _TASK_HANDLER.stream.flush()
            finally:
                _TASK_HANDLER.release()

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
