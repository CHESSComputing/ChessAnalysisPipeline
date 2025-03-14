"""FOXDEN utils module."""

# system modules
import os

# 3rd party modules
import requests


def HttpRequest(
        self, data, url, method='GET', headers=None,
        tokenEnv='CHESS_READER_TOKEN', timeout=10, dryRun=False):
    """Read data from FOXDEN service

    :param data: Input data.
    :type data: list[PipelineData]
    :param url: URL of service.
    :type url: str
    :param method: HTTP method to use, `"POST"` for creation and
        `"PUT"` for update, defaults to `"POST"`.
    :type method: str, optional
    :param headers: HTTP headers to use.
    :type headers: dictionary, optional
    :param tokenEnv: environment token variable
    :type tokenEnv: string
    :param timeout: Timeout of HTTP request, defaults to `10`.
    :type timeout: str, optional
    :param dryRun: `dryRun` option to verify HTTP workflow,
        defaults to `False`.
    :type dryRun: bool, optional
    :return: Contents of the input data.
    :rtype: object
    """
    if headers is None:
        headers = {}
    if 'Content-Type' not in headers:
        headers['Content-type'] = 'application/json'
    if 'Accept' not in headers:
        headers['Accept'] = 'application/json'
    if dryRun:
        print('### HTTP reader call', url, headers, data)
        return []
    token = ''
    fname = os.getenv(tokenEnv)
    if not fname:
        raise Exception(f'{tokenEnv} env variable is not set')
    with open(fname, 'r') as istream:
        token = istream.read()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    else:
        raise Exception(
                f'Valid write token missing in {tokenEnv} env variable')

    # Make actual HTTP request to FOXDEN service
    if method.lower() == 'post':
        resp = requests.get(
            url, headers=headers, timeout=timeout)
    elif method.lower() == 'post':
        resp = requests.post(
            url, headers=headers, timeout=timeout, data=data)
    elif method.lower() == 'put':
        resp = requests.put(
            url, headers=headers, timeout=timeout, data=data)
    elif method.lower() == 'delete':
        resp = requests.delete(
            url, headers=headers, timeout=timeout, data=data)
    else:
        raise Exception(f'unsupported method {method}')
    data = resp.content
    return data
