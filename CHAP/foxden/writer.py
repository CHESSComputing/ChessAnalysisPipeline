#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: FOXDEN writers
"""

# System modules
import json
from typing import Optional

# Third party modules
from pydantic import constr

# Local modules
from CHAP.pipeline import PipelineItem
from CHAP.foxden.utils import HttpRequest

class FoxdenDoiWriter(PipelineItem):
    """Writer for saving info to the FOXDEN DOI service."""
    def write(
            self, url, data, provider='Datacite', description='', draft=True,
            publishMetadata=True, verbose=False):
        """Write data to the FOXDEN DOI service.

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param provider: DOI provider name, e.g. Zenodo, Datacite,
            Materialcommons, defaults to `'Datacite'`.
        :type provider: str, optional
        :param description: Dataset description.
        :type description: str, optional
        :param draft: Draft DOI flag, defaults to `True`.
        :type draft: bool, optional
        :param publishMetadata: Publish metadata with DOI,
            defaults to `True`.
        :type publishMetadata: bool, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: HTTP response from FOXDEN DOI service.
        :rtype: list[dict]
        """
        self.logger.info(
            f'Executing "process" with url={url} data={data}')
        rurl = f'{url}/publish'
        raise NotImplementedError
        # FIX it would be useful to perform validation of data
#        if isinstance(data, list) and len(data) == 1:
#            data = data[0]['data'][0]
#        if not isinstance(data, dict):
#            raise ValueError(f'Invalid "data" parameter ({data})')
        draft_str = 'on' if draft else ''
        publish_meta = 'on' if publishMetadata else ''
        payload = {
#            'did': did,
            'provider': provider.lower(),
            'draft': draft_str,
            'metadata': publish_meta,
            'description': description,
        }
        if verbose:
            self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='write')
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        result = [{'code': response.status_code, 'data': response.text}]
        return result


class FoxdenMetadataWriter(PipelineItem):
    """Writer for saving data to the FOXDEN Metadata service.

    :param url: URL of service.
    :type url: str
    :param verbose: Verbose output flag, defaults to `False`.
    :type verbose: bool, optional
    """
    url: constr(strict=True, strip_whitespace=True)
    verbose: Optional[bool] = None

    def write(self, data):
        """Write data to the FOXDEN Metadata service.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: HTTP response from FOXDEN Metadata service.
        :rtype: list[dict]
        """
        # System modules
        from getpass import getuser

        record = self.get_data(data, schema='metadata')
        if not isinstance(record, dict):
            raise ValueError('Invalid metadata record {(record)}')

        # FIX it would be useful to perform validation of record

        # Submit HTTP request and return response
        schema = record.pop('schema', 'Analysis')
        payload = json.dumps({'Schema': schema, 'Record': record})
        self.logger.info(f'method=POST url={self.url} payload={payload}')
        response = HttpRequest(self.url, payload, method='POST', scope='write')
        if self.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = [{'code': response.status_code, 'data': response.text}]
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            self.logger.warning(f'HTTP response:\n{response.__dict__}')
            result = []
        return result


class FoxdenProvenanceWriter(PipelineItem):
    """Writer for saving data to the FOXDEN Provenance service.

    :param url: URL of service.
    :type url: str
    :param verbose: Verbose output flag, defaults to `False`.
    :type verbose: bool, optional
    """
    url: constr(strict=True, strip_whitespace=True)
    verbose: Optional[bool] = None

    def write(self, data):
        """Write data to the FOXDEN Provenance service.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: HTTP response from FOXDEN Provenance service.
        :rtype: list[dict]
        """
        record = self.get_data(data, name='FoxdenProvenanceProcessor')
        if not isinstance(record, dict):
            raise ValueError('Invalid provenance record {(record)}')

        # FIX it would be useful to perform validation of data

        # Submit HTTP request and return response
        url = f'{self.url}/dataset'
        payload = json.dumps(record)
        self.logger.info(f'method=POST url=url, payload={payload}')
        response = HttpRequest(url, payload, method='POST', scope='write')
        if self.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = [{'code': response.status_code, 'data': response.text}]
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            self.logger.warning(f'HTTP response:\n{response.__dict__}')
            result = []
        return result


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
