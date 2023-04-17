#!/usr/bin/env python
'''
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific workflows.
'''

# system modules
import argparse
import inspect
import json
import logging
import sys
from time import time

# local modules
from CHAP import Reader

class BinaryFileReader(Reader):
    def _read(self, filename):
        '''Return a content of a given file name

        :param filename: name of the binart file to read from
        :return: the content of `filename`
        :rtype: binary
        '''
        with open(filename, 'rb') as file:
            data = file.read()
        return(data)

class MultipleReader(Reader):
    def read(self, readers, **_read_kwargs):
        '''Return resuts from multiple `Reader`s.

        :param readers: a dictionary where the keys are specific names that are
            used by the next item in the `Pipeline`, and the values are `Reader`
            configurations.
        :type readers: list[dict]
        :return: The results of calling `Reader.read(**kwargs)` for each item
            configured in `readers`.
        :rtype: list[dict[str,object]]
        '''

        t0 = time()
        self.logger.info(f'Executing "read" with {len(readers)} Readers')

        data = []
        for reader_config in readers:
            reader_name = list(reader_config.keys())[0]
            reader_class = getattr(sys.modules[__name__], reader_name)
            reader = reader_class()
            reader_kwargs = reader_config[reader_name]

            # Combine keyword arguments to MultipleReader.read with those to the reader
            # giving precedence to those in the latter
            combined_kwargs = {**_read_kwargs, **reader_kwargs}
            data.extend(reader.read(**combined_kwargs))

        self.logger.info(f'Finished "read" in {time()-t0:.3f} seconds\n')

        return(data)

class NexusReader(Reader):
    def _read(self, filename, nxpath='/'):
        '''Return the NeXus object stored at `nxpath` in the nexus file
        `filename`.

        :param filename: name of the NeXus file to read from
        :type filename: str
        :param nxpath: path to a specific loaction in the NeXus file to read
            from, defaults to `'/'`
        :type nxpath: str, optional
        :raises nexusformat.nexus.NeXusError: if `filename` is not a NeXus
            file or `nxpath` is not in `filename`.
        :return: the NeXus structure indicated by `filename` and `nxpath`.
        :rtype: nexusformat.nexus.NXobject
        '''

        from nexusformat.nexus import nxload

        nxobject = nxload(filename)[nxpath]
        return(nxobject)

class URLReader(Reader):
    def _read(self, url, headers={}):
        '''Make an HTTPS request to the provided URL and return the results.
        Headers for the request are optional.

        :param url: the URL to read
        :type url: str
        :param headers: headers to attach to the request, defaults to `{}`
        :type headers: dict, optional
        :return: the content of the response
        :rtype: object
        '''

        import requests

        resp = requests.get(url, headers=headers)
        data = resp.content

        self.logger.debug(f'Response content: {data}')

        return(data)

class YAMLReader(Reader):
    def _read(self, filename):
        '''Return a dictionary from the contents of a yaml file.

        :param filename: name of the YAML file to read from
        :return: the contents of `filename`
        :rtype: dict
        '''

        import yaml

        with open(filename) as file:
            data = yaml.safe_load(file)
        return(data)

if __name__ == '__main__':
    from CHAP.reader import main
    main()
