#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: FOXDEN writers
"""

# System modules
from copy import deepcopy
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

    :ivar url: URL of service.
    :vartype url: str
    :ivar verbose: Verbose output flag, defaults to `False`.
    :vartype verbose: bool, optional
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

        record = deepcopy(self.get_data(
            data, schema='foxden.reader.FoxdenMetadataReader', remove=False))
        if not isinstance(record, dict):
            self.logger.warning(
                'Invalid or unavailable metadata record {(record)}')
            return

        # FIX it would be useful to perform validation of record

        # Submit HTTP request and return response
        schema = record.pop('schema', 'Analysis')
        #if schema not in ('user', 'Composite'):
        #    record.pop('parent_did', None)
        payload = json.dumps({'Schema': schema, 'Record': record})
        self.logger.info(f'method=POST url={self.url} payload={payload}')
        response = HttpRequest(self.url, payload, method='POST', scope='write')
        if self.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            self.logger.info(f'Successfully submitted metadata record')
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            self.logger.warning(
                f'HTTP response:\n{response.__dict__}\npayload:\n{payload}')
        return {'status': response.status_code, 'response': response.text}


class FoxdenProvenanceWriter(PipelineItem):
    """Writer for saving data to the FOXDEN Provenance service.

    :ivar url: URL of service.
    :vartype url: str
    :ivar verbose: Verbose output flag, defaults to `False`.
    :vartype verbose: bool, optional
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
        # Local modules
        from CHAP.common.utils import (
            osinfo,
            environments,
        )
        from CHAP.pipeline import PipelineData

        provenance = self.get_data(
            data, schema='foxden.reader.FoxdenProvenanceReader')
        if not isinstance(provenance, dict):
            self.logger.warning(
                'Invalid or unavailable provenance provenance {(provenance)}')
            return

        # FIX it would be useful to perform validation of data

        # Add system info to provenance data
        record = deepcopy(provenance)
        record.update({
            'environments': environments(),
            'osinfo': osinfo(),
            'processing': 'CHAP pipeline',
            'scripts': [
                {'name': 'CHAP', 'parent_script': None, 'order_idx': 1}],
            'site': 'Cornell',
        })

        # Submit HTTP request and return response
        url = f'{self.url}/dataset'
        payload = json.dumps(record)
        self.logger.info(f'method=POST url=url, payload={payload}')
        response = HttpRequest(url, payload, method='POST', scope='write')
        if self.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            self.logger.info(f'Successfully submitted provenance record')
            provenance['parent_did'] = provenance.pop('did')
            provenance.pop('input_files', None)
            provenance.pop('output_files', None)
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            self.logger.warning(f'HTTP response:\n{response.__dict__}')
        result = {'status': response.status_code, 'response': response.text}
        return (PipelineData(name=self.name, data=result),
                PipelineData(
                    name=self.name, data=provenance,
                    schema='foxden.reader.FoxdenProvenanceReader'))


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
