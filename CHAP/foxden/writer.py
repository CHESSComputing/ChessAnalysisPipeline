#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: FOXDEN writers
"""

# System modules
import json

# Local modules
from CHAP.foxden.utils import HttpRequest
from CHAP.writer import Writer


class FoxdenDoiWriter(Writer):
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
        form_data = {
            'did': did,
            'provider': provider.lower(),
            'draft': draft_str,
            'metadata': publish_meta,
            'description': description,
        }
        if verbose:
            self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, form_data, method='POST', scope='write')
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        result = [{'code': response.status_code, 'data': response.text}]
        return result


class FoxdenMetadataWriter(Writer):
    """Writer for saving data to the FOXDEN Metadata service."""
    def write(self, data, url):#config=None):
        """Write data to the FOXDEN Metadata service.

        :param data: Input data.
        :type data: list[PipelineData]
        :param config: FOXDEN HTTP request configuration.
        :type config: CHAP.foxden.models.FoxdenRequestConfig
        :return: HTTP response from FOXDEN Metadata service.
        :rtype: list[dict]
        """
        # System modules
        from getpass import getuser

        record = self.get_data(data, schema='metadata')
        if not isinstance(record, dict):
            raise ValueError('Invalid metadata record {(record)}')

        # FIX it would be useful to perform validation of record

        # Load and validate the FoxdenRequestConfig configuration
        config = self.get_config(
            config={'url': url}, schema='foxden.models.FoxdenRequestConfig')
#            config=config, schema='foxden.models.FoxdenRequestConfig')
        self.logger.debug(f'config: {config}')

        # For now cut out anything but the did and application fields
        # from the CHAP workflow metadata record
        record = {'did': record['did'],
                  'application': record.get('application', 'CHAP'),
                  'btr': record.get('btr'),
                  'user': getuser()}

        # Submit HTTP request and return response
        rurl = f'{config.url}'
        mrec = {'Schema': 'Analysis', 'Record': record}
        payload = json.dumps(mrec)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='write')
        if config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = [{'code': response.status_code, 'data': response.text}]
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            self.logger.warning(f'HTTP response:\n{response.__dict__}')
            result = []
        return result


class FoxdenProvenanceWriter(Writer):
    """Writer for saving data to the FOXDEN Provenance service."""
    def write(self, data, url):#config=None):
        """Write data to the FOXDEN Provenance service.

        :param data: Input data.
        :type data: list[PipelineData]
        :param config: FOXDEN HTTP request configuration.
        :type config: CHAP.foxden.models.FoxdenRequestConfig
        :return: HTTP response from FOXDEN Provenance service.
        :rtype: list[dict]
        """
        record = self.get_data(data, name='FoxdenProvenanceProcessor')
        if not isinstance(record, dict):
            raise ValueError('Invalid provenance record {(record)}')

        # FIX it would be useful to perform validation of data

        # Load and validate the FoxdenRequestConfig configuration
        config = self.get_config(
            config={'url': url}, schema='foxden.models.FoxdenRequestConfig')
#            config=config, schema='foxden.models.FoxdenRequestConfig')
        self.logger.debug(f'config: {config}')

        # Submit HTTP request and return response
        rurl = f'{url}/dataset'
        payload = json.dumps(record)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='write')
        if config.verbose:
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
