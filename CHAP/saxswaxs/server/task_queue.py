"""Queueing system for data processing tasks"""

import queue
import threading
from time import sleep, time
import traceback

from CHAP.saxswaxs.server.logging_config import (
    get_logger,
    task_log_context,
)
from CHAP.saxswaxs.server.slack import send_slack_message

TRY_N = 3
RETRY_SLEEP_T = 3

logger = get_logger('task_queue')

_task_queue = queue.Queue()

def _worker():
    """Continuously dequeue and execute tasks from the task queue.

    Runs in a background daemon thread. On task failure the error is logged
    and the task is marked done.
    """
    while True:
        task, args, kwargs, log_path = _task_queue.get()
        logger.info(
            f'Starting task: {str(task)}, args: {args}, kwargs: {kwargs}')
        t0 = time()
        success = False
        try_n = 1
        while not success and try_n <= TRY_N:
            try_n += 1
            with task_log_context(log_path):
                # Handle race conditions from missing data
                try:
                    task(*args, **kwargs)
                    success = True
                except Exception as exc:
                    if try_n <= TRY_N:
                        sleep(RETRY_SLEEP_T)
                        continue
                    else:
                        traceback_text = traceback.format_exc()

                        logger.error(f'Task failed: {exc}')
                        print(traceback_text)

                        try:
                            send_slack_message(
                                f'*Task failed:* `{task}`\n'
                                f'*args:* `{args}`\n'
                                f'*kwargs:* `{kwargs}`\n'
                                f'```{traceback_text}```'
                            )
                        except Exception:
                            logger.exception('Failed to send Slack notification')
                        break
        _task_queue.task_done()
        tf = time()
        logger.info(f'Task done. ({tf-t0:.5f} seconds)')

def put(task, args, kwargs):
    """Enqueue a task for execution by the background worker thread.

    :param task: Callable to execute.
    :param args: Positional arguments to pass to ``task``.
    :type args: tuple
    :param kwargs: Keyword arguments to pass to ``task``.
    :type kwargs: dict
    """
    cfg = args[0]
    if hasattr(cfg, 'outputdir'):
        log_path = cfg.outputdir / f"chap_{task.__name__}.log"
    else:
        log_path = cfg.data_zarr.parent / f"chap_{task.__name__}.log"
    _task_queue.put((task, args, kwargs, log_path))

threading.Thread(target=_worker, daemon=True).start()
