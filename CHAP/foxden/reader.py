#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: FOXDEN readers
"""

# System modules
import json
from time import time

# Third party modules
import requests

# Local modules
from CHAP.foxden.utils import HttpRequest
from CHAP.reader import Reader


class FoxdenDataDiscoveryReader(Reader):
    """Reader for the FOXDEN Data Discovery service."""
    def read(
            self, url, did='', query='', idx=0, limit=10, verbose=False):
        """Read records from the FOXDEN Data Discovery service based on
        did or an arbitrary query.

        :param url: URL of service.
        :type url: str
        :param did: FOXDEN dataset identifier (did).
        :type did: string, optional
        :param query: FOXDEN query.
        :type query: string, optional
        :param idx: Ask Valentin
        :type idx: int, optional
        :param limit: Maximum number of returned records,
            defaults to `10`.
        :type limit: int, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        self.logger.info(f'url: {url}')
        self.logger.info(f'did: {did}')

        self.logger.info(f'query: {query}')
        rurl = f'{url}/search'
        request = {'client': 'CHAP-FoxdenDataDiscoveryReader',
                   'service_query': {'idx': idx, 'limit': limit}}
        if did:
            request['service_query'].update({'did': did})
        if query:
            request['service_query'].update({'query': query})
        payload = json.dumps(request)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='read')
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            data = json.loads(response.text)
#            for k, v in data.items():
#                print(f'\t{k}: {type(v)}')
#            nrecords = data['results']['nrecords']
#            records = data['results']['records']
#            print(f'\n\nnrecords: {nrecords}')
#            print(f'\nrecords[0] {type(records[0])}: {records[0]}\n\n')
            self.logger.debug(
                f'Found a total of {data["results"]["nrecords"]} records')
            data = data['results']['records']
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            data = []
        self.logger.debug(
            f'Returning {len(data)} records')
        return data


class FoxdenMetadataReader(Reader):
    """Reader for FOXDEN Metadata data from a specific FOXDEN Metadata
    service.
    """
    def read(
            self, url, data, did='', query='', spec=None, method='POST', #'GET',
            verbose=False):
#            headers=None, scope='read', dry_run=False, verbose=False):
        """Read and return data from a specific FOXDEN Metadata service.

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (did).
        :type did: string, optional
        :param query: FOXDEN query.
        :type query: string, optional
        :param spec: FOXDEN spec.
        :type spec: dictionary, optional
        :param method: HTTP method to use, `'POST'` for creation or
            `'PUT'` for update, defaults to `'POST'`.
        :type method: str, optional
        :param headers: HTTP headers to use.
        :type headers: dictionary, optional
        :param scope: FOXDEN scope: `'read'` or `'write'`.
        :type scope: string
        :param dry_run: `dry_run` option to verify HTTP workflow,
            defaults to `False`.
        :type dry_run: bool, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/search'
        request = {'client': 'CHAP-FoxdenMetadataReader', 'service_query': {}}
        if did:
            request['service_query'].update({'spec': {'did': did}})
        if query:
            request['service_query'].update({'query': query})
        if spec:
            request['service_query'].update({'spec': spec})
        payload = json.dumps(request)
        if verbose:
            self.logger.info(f'method={method} url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method=method)
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            data = json.loads(response.text)
        else:
            data = []
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data

class FoxdenProvenanceReader(Reader):
    """Reader for FOXDEN Provenance data from a specific FOXDEN
    Provenance service.
    """
    def read(
            self, url, data, did='', method='POST', verbose=False):
            #self, url, data, did='', method='GET', verbose=False):
        """Read data from a specific FOXDEN Provenance service and
        return the file names to the upstream caller.

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (did).
        :type did: string, optional
        :param method: HTTP method to use, `'POST'` for creation or
            `'PUT'` for update, defaults to `'POST'`.
        :type method: str, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: List of file names from the FOXDEN provenance service.
        :rtype: list
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/files?did={did}'
        payload = None
        if verbose:
            self.logger.info(f'method={method} url={rurl}')
        response = HttpRequest(rurl, payload, method=method)
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            data = []
            # Receive FOXDEN provenance records and extract only the
            # file names to return to the upstream caller
            for rec in json.loads(response.text):
                data.append(rec['name'])
        else:
            data = []
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data

class FoxdenSpecScansReader(Reader):
    """Reader for FOXDEN SpecScans data from a specific FOXDEN
    SpecScans service.
    """
    def read(
            self, url, data, did='', query='', spec=None, method='POST', #'GET',
            verbose=False):
        """Read and return data from a specific FOXDEN SpecScans
        service.

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (did).
        :type did: string, optional
        :param query: FOXDEN query.
        :type query: string, optional
        :param spec: FOXDEN spec.
        :type spec: dictionary, optional
        :param method: HTTP method to use, `'POST'` for creation or
            `'PUT'` for update, defaults to `'POST'`.
        :type method: str, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/search'
        request = {'client': 'CHAP-FoxdenSpecScansReader', 'service_query': {}}
        if did:
            request['service_query'].update({'spec': {'did': did}})
        if query:
            request['service_query'].update({'query': query})
        if spec:
            request['service_query'].update({'spec': spec})
        payload = json.dumps(request)
        if verbose:
            self.logger.info(f'method={method} url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method=method)
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
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
