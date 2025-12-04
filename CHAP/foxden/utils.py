"""FOXDEN utils module."""


def readFoxdenToken(scope):
    """Obtain a FOXDEN token.

    :param scope: FOXDEN scope: `'read'` or `'write'`.
    :type scope: string
    """
    # System modules
    import os

    token = os.getenv(f'FOXDEN_{scope.upper()}_TOKEN')
    if not token:
        token_file = os.path.join(
            os.getenv('HOME'), f'.foxden.{scope.lower()}.token')
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                token = f.read()
    return token

def HttpRequest(
        url, payload, method='POST', headers=None, scope='read', timeout=10,
        dry_run=False):
    """Submit a HTTP request to a FOXDEN service

    :param url: URL of service.
    :type url: str
    :param payload: HTTP request payload.
    :type payload: str
    :param method: HTTP method to use, defaults to `'POST'`.
    :type method: str, optional
    :param headers: HTTP headers to use.
    :type headers: dictionary, optional
    :param scope: FOXDEN scope: `'read'` or `'write'`,
        defaults to `'read'`.
    :type scope: string, optional
    :param timeout: Timeout of HTTP request, defaults to `10`.
    :type timeout: str, optional
    :param dry_run: `dry_run` option to verify HTTP workflow,
        defaults to `False`.
    :type dry_run: bool, optional
    :return: HTTP response.
    :rtype: requests.models.Response
    """
    # Third party modules
    from requests import (
        get,
        post,
        put,
        delete,
    )

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
        raise RuntimeError(f'Unable to obtain token with scope {scope}')
    if dry_run:
        print('### HTTP reader call', url, headers, payload)
        return []

    # Make actual HTTP request to FOXDEN service
    if method.lower() == 'get':
        response = get(url, headers=headers, timeout=timeout)
    elif method.lower() == 'post':
        response = post(url, headers=headers, timeout=timeout, data=payload)
    elif method.lower() == 'put':
        response = put(url, headers=headers, timeout=timeout, data=payload)
    elif method.lower() == 'delete':
        response = delete(url, headers=headers, timeout=timeout, data=payload)
    else:
        raise ValueError(f'Unsupported method {method}')
    return response
