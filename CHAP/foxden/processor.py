#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module for FOXDEN services
"""

# System modules
from time import time
import json

# Local modules
from CHAP.processor import Processor
from CHAP.foxden.writer import FoxdenWriter
from CHAP.foxden.utils import HttpRequest

class FoxdenMetaDataProcessor(Processor):
    """A Processor to communicate with FOXDEN MetaData server."""

    def process(self, data, url, did, dryRun=False, verbose=False):
        """FOXDEN MetaData processor

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param did: FOXDEN dataset identifier (did)
        :type did: string
        :param dryRun: `dryRun` option to verify HTTP workflow,
            defaults to `False`.
        :type dryRun: bool, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: data from FOXDEN MetaData service
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/search'
        request = {"client": "CHAP",
                   "service_query": {"query": "", "spec": {"did":did},
                                     "sql": "", "idx": 0, "limit": 0}}
        payload = json.dumps(request)
        response = HttpRequest(rurl, payload, method='POST')
        print("responpse", response, type(response))
        if response.status_code == 200:
            data = json.loads(response.text)
        else:
            data = []
        print("record\n", data, type(data))
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data

class FoxdenProvenanceProcessor(Processor):
    """A Processor to communicate with FOXDEN provenance server."""
    def process(self, data, url, did, dryRun=False, verbose=False):
        """FOXDEN Provenance processor

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param did: FOXDEN dataset identifier (did)
        :type did: string
        :param dryRun: `dryRun` option to verify HTTP workflow,
            defaults to `False`.
        :type dryRun: bool, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: data from FOXDEN provenance service
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/files?did={did}'
        payload = None
        response = HttpRequest(rurl, payload, method='GET')
        print("responpse", response, type(response))
        if response.status_code == 200:
            data = []
            # here we received FOXDEN provenance records
            # we will extract from them only file names
            # and return to upstream caller
            for rec in json.loads(response.text):
                data.append(rec["name"])
        else:
            data = []
        print("record\n", data, type(data))
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
