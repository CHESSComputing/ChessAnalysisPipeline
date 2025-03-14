"""FOXDEN utils module."""

# system modules
import os

# 3rd party modules
import requests


def readFoxdenToken(scope):
    """
    """
    token = ''
    rfile = os.path.join(os.getenv('HOME'), '.foxden.read.yaml')
    wfile = os.path.join(os.getenv('HOME'), '.foxden.write.yaml')
    if scope == 'read':
        if os.getenv('FOXDEN_READ_TOKEN'):
            token = os.getenv('FOXDEN_READ_TOKEN')
        elif os.path.exists(rfile):
            tfile = rfile
    elif scope == 'write':
        if os.getenv('FOXDEN_WRITE_TOKEN'):
            token = os.getenv('FOXDEN_WRITE_TOKEN')
        elif os.path.exists(rfile):
            tfile = wfile
    if not token and os.path.exists(tfile):
        with open(tfile, 'r') as istream:
            token = istream.read()
    return token

def HttpRequest(
        data, url, method='GET', headers=None,
        scope='read', timeout=10, dryRun=False):
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
    :param scope: FOXDEN scope to use, e.g. read or write
    :type scope: string
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
    token = readFoxdenToken(scope)
    if token:
        headers['Authorization'] = f'Bearer {token}'
    else:
        raise Exception(
                f'Unable to obtain token with scope {scope}')
    if dryRun:
        print('### HTTP reader call', url, headers, data)
        return []

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
