"""Module for interacting with Slack for monitoring"""


import os
import requests


SLACK_WEBHOOK_URL = os.environ['SLACK_WEBHOOK_URL']


def send_slack_message(message):
    """Send a message to Slack via an incoming webhook."""
    response = requests.post(
        SLACK_WEBHOOK_URL,
        json={'text': message},
        timeout=10,
    )
    response.raise_for_status()
