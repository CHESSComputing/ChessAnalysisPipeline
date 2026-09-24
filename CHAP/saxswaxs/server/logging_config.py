"""Logging configuration"""

import contextlib
import logging
import os
import sys
import time as _time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


_LOGDIR = None
_LOG_HANDLER = None


def set_logdir(path):
    global _LOGDIR, _LOG_HANDLER
    os.makedirs(path, exist_ok=True)
    _LOGDIR = path
    _LOG_HANDLER = None  # force re-creation against the new path
    new_handler = _get_log_handler()
    for lgr in logging.Logger.manager.loggerDict.values():
        if isinstance(lgr, logging.Logger) and not lgr.propagate:
            lgr.handlers = [new_handler]


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

    # Redirect stdout/stderr once a real file handler is available.
    if _STDOUT_WRAPPER is None and not isinstance(handler, logging.NullHandler):
        _STDOUT_WRAPPER = StreamToLogFile(handler)
        _STDERR_WRAPPER = StreamToLogFile(handler)

        sys.stdout = _STDOUT_WRAPPER
        sys.stderr = _STDERR_WRAPPER

    return logger


def get_task_logger(name, log_level="DEBUG"):
    """Return a logger that writes only to the active task log file.

    Unlike :func:`get_logger`, the returned logger is not attached to the
    shared application log handler.  If no task context is active the logger
    has no handlers and its output is silently discarded.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = [_TASK_HANDLER] if _TASK_HANDLER is not None else []
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

    # CHAP internal processors use StreamHandler(sys.stderr) loggers.
    # sys.stderr is _STDERR_WRAPPER (a StreamToLogFile), so their output
    # reaches the main log via StreamToLogFile.write().  Suppress that path
    # for the duration of the task so those messages go only to the task log.
    if _STDERR_WRAPPER is not None:
        _STDERR_WRAPPER._task_only = True

    try:
        yield
    finally:
        if _STDERR_WRAPPER is not None:
            _STDERR_WRAPPER._task_only = False

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
        self._task_only = False

    def write(self, message):
        if not message:
            return

        # Click and some other libraries may write bytes rather than str.
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")

        if not self._task_only:
            self.handler.acquire()
            try:
                # Rotate if necessary.
                record = logging.LogRecord(
                    name="stream",
                    level=logging.DEBUG,
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


class _DatestampedRotatingHandler(TimedRotatingFileHandler):
    """Like :class:`TimedRotatingFileHandler` but the active log file always
    carries a date stamp (``<base>.YYYY-MM-DD.log``).

    The standard handler writes to an undated ``<base>`` file and only adds a
    date suffix to the *rotated-away* copy.  This subclass inverts that: every
    file, including the first one opened on startup, is date-stamped.  On
    rollover the current file is left in place and a new dated file is opened
    for the incoming day; no renaming takes place.
    """

    def __init__(self, base_path, backup_count=30, encoding='utf-8'):
        self._log_base = base_path

        os.makedirs(os.path.dirname(self._log_base), exist_ok=True)

        super().__init__(
            filename=self._current_path(),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding=encoding,
        )

    def _current_path(self):
        from datetime import date
        return f"{self._log_base}_{date.today():%Y-%m-%d}.log"

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        if self.backupCount > 0:
            base_dir = os.path.dirname(os.path.abspath(self._log_base))
            base_name = os.path.basename(self._log_base)
            dated_files = sorted(
                os.path.join(base_dir, f)
                for f in os.listdir(base_dir)
                if f.startswith(base_name + '.') and f.endswith('.log')
            )
            for old in dated_files[:-self.backupCount]:
                try:
                    os.remove(old)
                except OSError:
                    pass

        self.baseFilename = os.path.abspath(self._current_path())
        self.stream = self._open()
        self.rolloverAt = self.computeRollover(int(_time.time()))


def _get_log_handler():
    global _LOGDIR
    global _LOG_HANDLER

    if _LOGDIR is None:
        return logging.NullHandler()

    if _LOG_HANDLER is None:
        log_file = os.path.join(
            _LOGDIR,
            "saxswaxs-server",
        )

        _LOG_HANDLER = _DatestampedRotatingHandler(
            log_file,
            backup_count=30,
            encoding="utf-8",
        )

        _LOG_HANDLER.setFormatter(
            logging.Formatter(
                "{asctime}: {name:20} (L{lineno}): "
                "{levelname}: {message}",
                datefmt="%Y-%m-%d %H:%M:%S",
                style="{",
            )
        )

    return _LOG_HANDLER
