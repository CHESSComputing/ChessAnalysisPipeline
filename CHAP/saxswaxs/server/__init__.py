"""Daemon-like application for efficient automated SAXS/WAXS data
processing."""

def get_logger(name=__name__, log_level="DEBUG"):
    """Create and return a :class:`logging.Logger` with a stream handler.

    Configures the logger with a formatted :class:`logging.StreamHandler`
    that writes to stderr.  Re-assigning ``logger.handlers`` ensures no
    duplicate handlers accumulate on repeated calls with the same *name*.

    :param name: Logger name, typically the calling module's ``__name__``.
    :type name: str
    :param log_level: Case-insensitive logging level string
        (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    :type log_level: str
    :returns: Configured logger instance.
    :rtype: logging.Logger
    """
    import logging

    logger = logging.getLogger(name)
    log_level = getattr(logging, log_level.upper())
    logger.setLevel(log_level)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        '{asctime}: {name:20} (L{lineno}): {levelname}: {message}',
        datefmt='%Y-%m-%d %H:%M:%S', style='{'))
    logger.addHandler(log_handler)
    logger.handlers = [log_handler]
    logger.propagate = False
    return logger
