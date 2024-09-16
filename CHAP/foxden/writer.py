"""FOXDEN command line writer."""

class FoxdenWriter():
    """FOXDEN writer writes data to specific FOXDEN service."""
    def write(
            self, data, url, method='POST', headers=None, timeout=10,
            dryRun=False):
        """Write the input data as text to a file.

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param method: HTTP method to use, `"POST"` for creation and
            `"PUT"` for update, defaults to `"POST"`.
        :type method: str, optional
        :param headers: HTTP headers to use.
        :type headers: dictionary, optional
        :param timeout: Timeout of HTTP request, defaults to `10`.
        :type timeout: str, optional
        :param dryRun: `dryRun` option to verify HTTP workflow,
            defaults to `False`.
        :type dryRun: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        # System modules
        import os

        # Third party modules
        import requests

        if headers is None:
            headers = {}
        if 'Content-Type' not in headers:
            headers['Content-type'] = 'application/json'
        if 'Accept' not in headers:
            headers['Accept'] = 'application/json'
        if dryRun:
            print('### HTTP writer call', url, headers, data)
            return []
        token = ''
        fname = os.getenv('CHESS_WRITE_TOKEN')
        if not fname:
            raise Exception(f'CHESS_WRITE_TOKEN env variable is not set')
        with open(fname, 'r') as istream:
            token = istream.read()
        if token:
            headers['Authorization'] = f'Bearer {token}'
        else:
            raise Exception(
                f'Valid write token missing in CHESS_WRITE_TOKEN env variable')

        # Make actual HTTP request to FOXDEN service
        if method.lower() == 'post':
            resp = requests.post(
                url, headers=headers, timeout=timeout, data=data)
        elif method.lower() == 'put':
            resp = requests.put(
                url, headers=headers, timeout=timeout, data=data)
        else:
            raise Exception(f'unsupporteed method {method}')
        data = resp.content
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
