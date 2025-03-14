"""FOXDEN writer."""

from CHAP.foxden.utils import HttpRequest


class FoxdenWriter():
    """FOXDEN writer writes data to specific FOXDEN service."""
    def write(
            self, data, url, method='POST', headers=None,
            tokenEnv='CHESS_WRITER_TOKEN', timeout=10, dryRun=False):
        """Write data to FOXDEN

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param method: HTTP method to use, `"POST"` for creation and
            `"PUT"` for update, defaults to `"POST"`.
        :type method: str, optional
        :param headers: HTTP headers to use.
        :param tokenEnv: environment token variable
        :type tokenEnv: string
        :type headers: dictionary, optional
        :param timeout: Timeout of HTTP request, defaults to `10`.
        :type timeout: str, optional
        :param dryRun: `dryRun` option to verify HTTP workflow,
            defaults to `False`.
        :type dryRun: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        return HttpRequest(data, url, method, headers, 'CHESS_READER_TOKEN', timeout, dryrun)


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
