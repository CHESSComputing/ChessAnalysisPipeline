"""Module for interacting with Slack for monitoring"""


import os
from pathlib import Path
import requests


def get_slack_webhook_url():
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if url:
        return url

    path = Path.home() / "config" / "chap_saxswaxs_server_slack_webhook_url"
    return path.read_text().strip()


SLACK_WEBHOOK_URL = get_slack_webhook_url()


def send_slack_message(message):
    """Send a message to Slack via an incoming webhook."""
    response = requests.post(
        SLACK_WEBHOOK_URL,
        json={'text': message},
        timeout=10,
    )
    response.raise_for_status()
