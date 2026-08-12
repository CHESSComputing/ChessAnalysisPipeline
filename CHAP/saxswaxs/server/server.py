"""Server for handling HTTP requests."""

from flask import Flask, jsonify, request
import logging
import time
from traceback import print_exc

from CHAP.saxswaxs.server import get_logger
from CHAP.saxswaxs.server.task_queue import put
from CHAP.saxswaxs.server.chap import (
    setup, update, convert, make_pipeline, convert_configs,
    SetupCfg, UpdateCfg, ConvertCfg, MakePipelineCfg, ConvertConfigsCfg,
)

app = Flask(__name__)
app.logger = get_logger('server')
app.logger.propagate = False

# Logging middleware
@app.before_request
def start_timer():
    """Record the request start time for duration logging."""
    request.start_time = time.time()

@app.after_request
def log_request(response):
    """Log the HTTP method, path, status code, and duration for the request."""
    now = time.time()
    duration = round(now - request.start_time, 4)
    app.logger.info(
        f"{request.method} {request.path} - {response.status_code} ({duration}s)"
    )
    return response

# API endpoints
@app.route('/setup', methods=['POST'])
def setup_handler():
    """Handle POST /setup — parse JSON body and queue a setup task."""
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify(
            {
                'status': 'error',
                'reason': 'no data in body of request'
            }
        ), 400
    try:
        cfg = SetupCfg(**body)
    except Exception as exc:
        print_exc()
        return (
            jsonify(
                {
                    'sattus': 'error',
                    'reason': repr(exc),
                }
            ),
            400
        )
    setup_args = (cfg,)
    setup_kwargs = {}
    put(setup, setup_args, setup_kwargs)
    return jsonify({'status': 'queued'}), 202

@app.route('/update', methods=['POST'])
def update_handler():
    """Handle POST /update — parse JSON body and queue an update task."""
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify(
            {
                'status': 'error',
                'reason': 'no data in body of request'
            }
        ), 400
    try:
        cfg = UpdateCfg(**body)
    except Exception as exc:
        print_exc()
        return (
            jsonify(
                {
                    'sattus': 'error',
                    'reason': repr(exc),
                }
            ),
            400
        )
    update_args = (cfg,)
    update_kwargs = {}
    put(update, update_args, update_kwargs)
    return jsonify({'status': 'queued'}), 202

@app.route('/convert', methods=['POST'])
def convert_handler():
    """Handle POST /convert — parse JSON body and queue a convert task."""
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify(
            {
                'status': 'error',
                'reason': 'no data in body of request'
            }
        ), 400
    try:
        cfg = ConvertCfg(**body)
    except Exception as exc:
        print_exc()
        return (
            jsonify(
                {
                    'sattus': 'error',
                    'reason': repr(exc),
                }
            ),
            400
        )
    convert_args = (cfg,)
    convert_kwargs = {}
    put(convert, convert_args, convert_kwargs)
    return jsonify({'status': 'queued'}), 202


@app.route('/make_pipeline', methods=['POST'])
def make_pipeline_handler():
    """Handle POST /make_pipeline — parse JSON body and queue a make_pipeline task.

    Constructs a :class:`~CHAP.saxswaxs.server.chap.MakePipelineCfg` from the request
    body and queues :func:`~CHAP.saxswaxs.server.chap.make_pipeline` to write a
    ``pipeline.yaml`` from the pre-existing config files in ``outputdir``.
    """
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify(
            {
                'status': 'error',
                'reason': 'no data in body of request'
            }
        ), 400
    try:
        cfg = MakePipelineCfg(**body)
    except Exception as exc:
        print_exc()
        return (
            jsonify(
                {
                    'status': 'error',
                    'reason': repr(exc),
                }
            ),
            400
        )
    put(make_pipeline, (cfg,), {})
    return jsonify({'status': 'queued'}), 202


@app.route('/convert_configs', methods=['POST'])
def convert_configs_handler():
    """Handle POST /convert_configs — parse JSON body and queue a convert_configs task.

    Constructs a :class:`~CHAP.saxswaxs.server.chap.ConvertConfigsCfg` from the request
    body and queues :func:`~CHAP.saxswaxs.server.chap.convert_configs` to write detector,
    pyFAI integration, and corrections config YAML files from the provided
    tool config files.
    """
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify(
            {
                'status': 'error',
                'reason': 'no data in body of request'
            }
        ), 400
    try:
        cfg = ConvertConfigsCfg(**body)
    except Exception as exc:
        print_exc()
        return (
            jsonify(
                {
                    'status': 'error',
                    'reason': repr(exc),
                }
            ),
            400
        )
    put(convert_configs, (cfg,), {})
    return jsonify({'status': 'queued'}), 202


def run():
    """Start the Flask development server."""
    app.run(debug=False)

if __name__ == '__main__':
    run()
