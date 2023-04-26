#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific
             workflows.
"""

# system modules
from sys import modules
from time import time

# local modules
from CHAP import Reader


class BinaryFileReader(Reader):
    """Reader for binary files"""
    def read(self, filename):
        """Return a content of a given file name

        :param filename: name of the binart file to read from
        :return: the content of `filename`
        :rtype: binary
        """

        with open(filename, 'rb') as file:
            data = file.read()
        return data


class NexusReader(Reader):
    """Reader for NeXus files"""
    def read(self, filename, nxpath='/'):
        """Return the NeXus object stored at `nxpath` in the nexus
        file `filename`.

        :param filename: name of the NeXus file to read from
        :type filename: str
        :param nxpath: path to a specific loaction in the NeXus file
            to read from, defaults to `'/'`
        :type nxpath: str, optional
        :raises nexusformat.nexus.NeXusError: if `filename` is not a
            NeXus file or `nxpath` is not in `filename`.
        :return: the NeXus structure indicated by `filename` and `nxpath`.
        :rtype: nexusformat.nexus.NXobject
        """

        from nexusformat.nexus import nxload

        nxobject = nxload(filename)[nxpath]
        return nxobject


class URLReader(Reader):
    """Reader for data available over HTTPS"""
    def read(self, url, headers={}, timeout=10):
        """Make an HTTPS request to the provided URL and return the
        results.  Headers for the request are optional.

        :param url: the URL to read
        :type url: str
        :param headers: headers to attach to the request, defaults to
            `{}`
        :type headers: dict, optional
        :return: the content of the response
        :rtype: object
        """

        import requests

        resp = requests.get(url, headers=headers, timeout=timeout)
        data = resp.content

        self.logger.debug(f'Response content: {data}')

        return data


class YAMLReader(Reader):
    """Reader for YAML files"""
    def read(self, filename):
        """Return a dictionary from the contents of a yaml file.

        :param filename: name of the YAML file to read from
        :return: the contents of `filename`
        :rtype: dict
        """

        import yaml

        with open(filename) as file:
            data = yaml.safe_load(file)
        return data


if __name__ == '__main__':
    from CHAP.reader import main
    main()
