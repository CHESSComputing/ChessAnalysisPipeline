#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: FOXDEN readers
"""

# system modules
from time import time
import json

# 3rd party modules
import requests

# CHAP modules
from CHAP.foxden.utils import HttpRequest
from CHAP.pipeline import PipelineItem


class FoxdenMetadataReader(PipelineItem):
    """FOXDEN Metadata reader reads data from specific FOXDEN Metadata service."""
    def read(
            self, url, data, did='', query='', spec=None,
            method='GET', headers=None,
            scope='read', dryRun=False, verbose=False):
        """Read data from FOXDEN service

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (did)
        :type did: string, optional
        :param query: FOXDEN query
        :type query: string, optional
        :param spec: FOXDEN spec
        :type spec: dictionary, optional
        :param method: HTTP method to use, `"POST"` for creation and
            `"PUT"` for update, defaults to `"POST"`.
        :type method: str, optional
        :param headers: HTTP headers to use.
        :type headers: dictionary, optional
        :param scope: FOXDEN scope to use, e.g. read or write
        :type scope: string
        :param dryRun: `dryRun` option to verify HTTP workflow,
            defaults to `False`.
        :type dryRun: bool, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/search'
        request = {"client": "CHAP-FoxdenMetadataReader", "service_query": {}}
        if did and did != "":
            request["service_query"].update({"spec": {"did": did}})
        if query and query != "":
            request["service_query"].update({"query": query})
        if spec:
            request["service_query"].update({"spec": spec})
        payload = json.dumps(request)
        if verbose:
            self.logger.info(f"method=POST url={rurl} payload={payload}")
        response = HttpRequest(rurl, payload, method='POST')
        if verbose:
            self.logger.info(f"code={response.status_code} data={response.text}")
        if response.status_code == 200:
            data = json.loads(response.text)
        else:
            data = []
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data

class FoxdenProvenanceReader(PipelineItem):
    """FOXDEN Provenance reader reads data from specific FOXDEN Provenance service."""
    def read(
            self, url, data, did='', method='GET', headers=None,
            scope='read', dryRun=False, verbose=False):
        """FOXDEN Provenance processor

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (did)
        :type did: string, optional
        :param query: FOXDEN query
        :type query: string, optional
        :param dryRun: `dryRun` option to verify HTTP workflow,
            defaults to `False`.
        :type dryRun: bool, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: data from FOXDEN provenance service
        :rtype: list of file names
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/files?did={did}'
        payload = None
        if verbose:
            self.logger.info(f"method=GET url={rurl}")
        response = HttpRequest(rurl, payload, method='GET')
        if verbose:
            self.logger.info(f"code={response.status_code} data={response.text}")
        if response.status_code == 200:
            data = []
            # here we received FOXDEN provenance records
            # we will extract from them only file names
            # and return to upstream caller
            for rec in json.loads(response.text):
                data.append(rec["name"])
        else:
            data = []
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data

class FoxdenSpecScansReader(PipelineItem):
    """FOXDEN SpecScans reader reads data from specific FOXDEN SpecScans service."""
    def read(
            self, url, data, did='', query='', spec=None,
            method='GET', headers=None,
            scope='read', dryRun=False, verbose=False):
        """Read data from FOXDEN service

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (did)
        :type did: string, optional
        :param query: FOXDEN query
        :type query: string, optional
        :param spec: FOXDEN spec
        :type spec: dictionary, optional
        :param method: HTTP method to use, `"POST"` for creation and
            `"PUT"` for update, defaults to `"POST"`.
        :type method: str, optional
        :param headers: HTTP headers to use.
        :type headers: dictionary, optional
        :param scope: FOXDEN scope to use, e.g. read or write
        :type scope: string
        :param dryRun: `dryRun` option to verify HTTP workflow,
            defaults to `False`.
        :type dryRun: bool, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/search'
        request = {"client": "CHAP-FoxdenSpecScansReader", "service_query": {}}
        if did != "":
            request["service_query"].update({"spec": {"did": did}})
        if query != "":
            request["service_query"].update({"query": query})
        if spec:
            request["service_query"].update({"spec": spec})
        payload = json.dumps(request)
        if verbose:
            self.logger.info(f"method=POST url={rurl} payload={payload}")
        response = HttpRequest(rurl, payload, method='POST')
        if verbose:
            self.logger.info(f"code={response.status_code} data={response.text}")
        if response.status_code == 200:
            data = json.loads(response.text)
        else:
            data = []
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.reader import main

    main()
