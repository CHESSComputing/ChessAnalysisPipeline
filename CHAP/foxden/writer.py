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
        :rtype: list with dictionary entry
        """
        self.logger.info(
            f'Executing "process" with url={url} data={data}')
        rurl = f'{url}/publish'
        # FIX it would be useful to perform validation of data
        if isinstance(data, list) and len(data) == 1:
            data = data[0]['data'][0]
        if not isinstance(data, dict):
            raise ValueError(f'Invalid "data" parameter ({data})')
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
        data = [{'code': response.status_code, 'data': response.text}]
        return data


class FoxdenMetadataWriter(Writer):
    """Writer for saving data to the FOXDEN Metadata service."""
    def write(self, url, data, method='POST', verbose=False):
        """Write data to the FOXDEN Metadata service.

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param method: HTTP method to use, `'POST'` for creation or
            `'PUT'` for update, defaults to `'POST'`.
        :type method: str, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: HTTP response from FOXDEN Metadata service.
        :rtype: list[dict]
        """
        self.logger.info(
            f'Executing "process" with url={url} data={data}')
        # FIX it would be useful to perform validation of data
        if isinstance(data, list) and len(data) == 1:
            data = data[0]['data'][0]
        if not isinstance(data, dict):
            raise ValueError(f'Invalid "data" parameter ({data})')
        mrec = {'Schema': 'Analysis', 'Record': data}
        payload = json.dumps(mrec)
        if verbose:
            self.logger.info(f'method={method} url={url} payload={payload}')
        response = HttpRequest(url, payload, method=method, scope='write')
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        data = [{'code': response.status_code, 'data': response.text}]
        return data


class FoxdenProvenanceWriter(Writer):
    """Writer for saving data to the FOXDEN Provenance service."""
    def write(self, data, url, verbose=False):
        """Write data to the FOXDEN Provenance service.

        :param data: Provenance data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: HTTP response from FOXDEN Provenance service.
        :rtype: list[dict]
        """
        self.logger.debug(f'url: {url}')
        self.logger.debug(f'data={data}')
        rurl = f'{url}/dataset'
        # FIX it would be useful to perform validation of data
#        print(f'\n\ndata {type(data)}:\n{data}\n\n')
#        ddata = self.unwrap_pipelinedata(data)[-1]
#        exit(f'\n\nddata {type(ddata)}:\n{ddata}\n\n')
        if isinstance(data, list) and len(data) == 1:
            data = data[0]['data'][0]
        if not isinstance(data, dict):
            raise ValueError(f'Invalid "data" parameter ({data})')
        payload = json.dumps(data)
        if verbose:
            self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='write')
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        data = [{'code': response.status_code, 'data': response.text}]
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
