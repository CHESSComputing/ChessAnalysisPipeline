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
    def read(self, config):
        """Read records from the FOXDEN Data Discovery service based on
        did or an arbitrary query.

        :param config: FOXDEN HTTP request configuration.
        :type config: CHAP.foxden.models.FoxdenRequestConfig
        :return: Contents of the input data.
        :rtype: object
        """
        # Load and validate the FoxdenRequestConfig configuration
        config = self.get_config(
            config=config, schema='foxden.models.FoxdenRequestConfig')
        self.logger.debug(f'config: {config}')

        # Submit HTTP request and return response
        rurl = f'{config.url}/search'
        request = {'client': 'CHAP-FoxdenDataDiscoveryReader'}
        if config.did is None:
            if config.query is None:
                query = '{}'
            else:
                query = config.query
            request['service_query'] = {'query': query, 'limit': config.limit}
        else:
            if config.limit is not None:
                self.logger.warning(
                    f'Ignoring parameter "limit" ({config.limit}), '
                    'when "did" is specified')
            if config.query is not None:
                self.logger.warning(
                    f'Ignoring parameter "query" ({config.query}), '
                    'when "did" is specified')
            request['service_query'] = {'query': f'did:{config.did}'}
        payload = json.dumps(request)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='read')
        if config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = json.loads(response.text)['results']['records']
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            result = []
        self.logger.debug(
            f'Returning {len(result)} records')
        return result


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
        :param spec: FOXDEN spec. ASK Valentin
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
