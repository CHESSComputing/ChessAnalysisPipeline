"""FOXDE command line writer."""

# system modules
import os

# Local modules
from CHAP.writer import main

class FoxdenWriter():
    """FOXDEN writer writes data to specific FOXDEN service
    """

    def write(self, data, url, method="POST", headers={}, timeout=10, dryRun=False):
        """Write the input data as text to a file.

        :param data: input data
        :type data: list[PipelineData]
        :param url: url of service
        :type url: str
        :param method: HTTP method to use, POST for creation and PUT for update
        :type method: str
        :param headers: HTTP headers to use
        :type headers: dictionary
        :param timeout: timeout of HTTP request
        :type timeout: str
        :param dryRun: dryRun option to verify HTTP workflow
        :type dryRun: boolean
        :return: contents of the input data
        :rtype: object
        """
        import requests
        if 'Content-Type' not in headers:
            headers['Content-type'] = 'application/json'
        if 'Accept' not in headers:
            headers['Accept'] = 'application/json'
        if dryRun:
            print("### HTTP writer call", url, headers, data)
            return []
        token = ""
        fname = os.getenv("CHESS_WRITE_TOKEN")
        if not fname:
            msg = f'CHESS_WRITE_TOKEN env variable is not set'
            raise Exception(msg)
        with open(fname, 'r') as istream:
            token = istream.read()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            msg = f'No valid write token found in CHESS_WRITE_TOKEN env variable'
            raise Exception(msg)

        # make actual HTTP request to FOXDEN service
        if method.lower() == 'post':
            resp = requests.post(url, headers=headers, timeout=timeout, data=data)
        elif method.lower() == 'put':
            resp = requests.put(url, headers=headers, timeout=timeout, data=data)
        else:
            msg = f"unsupporteed method {method}"
            raise Exception(msg)
        data = resp.content
        return data


if __name__ == '__main__':
    main()
