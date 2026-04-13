#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module for Readers unique to the
`FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
integration with CHAP.
"""

# System modules
import json
from typing import Optional

# Third party modules
from pydantic import (
    Field,
    model_validator,
)

# Local modules
from CHAP.foxden.models import FoxdenRequestConfig
from CHAP.foxden.utils import HTTP_request
from CHAP.pipeline import PipelineItem
from CHAP.processor import Processor


class FoxdenDataDiscoveryReader(PipelineItem):
    """Reader for the
    `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
    Data Discovery service.

    :ivar config: Initialization parameters for an instance of
        :py:class`~CHAP.foxden.models.FoxdenRequestConfig`.
    :vartype config: dict, optional
    """
    pipeline_fields: dict = Field(
        default = {'config': 'foxden.models.FoxdenRequestConfig'},
        init_var=True)
    config: Optional[FoxdenRequestConfig] = FoxdenRequestConfig()

    _validate_config = model_validator(mode='before')(
        Processor.validate_processor_before)

    @model_validator(mode='after')
    def validate_foxdendatadiscoveryreader_after(self):
        """Validate the model configuration.

        :return: Validated model configuration
        :rtype: dict
        """
        assert self.config.url is not None
        return self

    def read(self):
        """Read records from the
        `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
        Data Discovery service based on
        DID (Dataset Identifier) or an arbitrary query.

        :return: Discovered data records.
        :rtype: list
        """
        self.logger.debug(f'config: {self.config}')

        # Submit HTTP request and return response
        rurl = f'{self.config.url}/search'
        payload = self.config.create_http_request_payload(self)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HTTP_request(rurl, payload, method='POST', scope='read')
        if self.config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = json.loads(response.text)
            self.logger.debug(f'Returning {len(result)} records')
            return result
        self.logger.warning(f'HTTP error code {response.status_code}')
        self.logger.warning(
            f'HTTP response:\n{response.__dict__}\npayload:\n{payload}')
        return []


class FoxdenMetadataReader(PipelineItem):
    """Reader for the
    `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
    Metadata service.

    :ivar config: Initialization parameters for an instance of
        :py:class`~CHAP.foxden.models.FoxdenRequestConfig`.
    :vartype config: dict, optional
    """
    pipeline_fields: dict = Field(
        default = {'config': 'foxden.models.FoxdenRequestConfig'},
        init_var=True)
    config: Optional[FoxdenRequestConfig] = FoxdenRequestConfig()

    _validate_config = model_validator(mode='before')(
        Processor.validate_processor_before)

    @model_validator(mode='after')
    def validate_foxdenmetadatareader_after(self):
        """Validate the model configuration.

        :return: Validated model configuration
        :rtype: dict
        """
        if self.get_schema() is None:
            self.schema_ = 'foxden.reader.FoxdenMetadataReader'
        assert self.config.url is not None
        return self

    def read(self):
        """Read a record from the
        `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
        Metadata service based on DID (Dataset Identifier) or an
        arbitrary query.

        :return: Metadata record.
        :rtype: dict
        """
        self.logger.debug(f'config: {self.config}')

        # Submit HTTP request and return response
        rurl = f'{self.config.url}/search'
        payload = self.config.create_http_request_payload(self)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HTTP_request(rurl, payload, method='POST', scope='read')
        if self.config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = json.loads(response.text)
            if not isinstance(result, list):
                self.logger.warning('FOXDEN Metadata service did not return a '
                                    f'valid record for did {self.config.did}.')
                return []
            if not result:
                self.logger.warning('FOXDEN Metadata service did not return '
                                    f'any records for did {self.config.did}.')
                return []
            if len(result) > 1:
                self.logger.debug(f'Received {len(result)} records')
                self.logger.warning('FOXDEN Metadata service did not return a '
                                    f'unique record for did {self.config.did}.'
                                    ' Returning the last record.')
            # FIX For now don't allow multiple parent did records
#            assert 'parent_did' not in self._metadata
#            self._metadata['parent_did'] = result[-1]['did']
            return result[-1]
        self.logger.warning(f'HTTP error code {response.status_code}')
        self.logger.warning(
            f'HTTP response:\n{response.__dict__}\npayload:\n{payload}')
        return []


class FoxdenProvenanceReader(PipelineItem):
    """Reader for the
    `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
    Provenance service.

    :ivar config: Initialization parameters for an instance of
        :py:class`~CHAP.foxden.models.FoxdenRequestConfig`.
    :vartype config: dict, optional
    """
    pipeline_fields: dict = Field(
        default = {'config': 'foxden.models.FoxdenRequestConfig'},
        init_var=True)
    config: Optional[FoxdenRequestConfig] = FoxdenRequestConfig()

    _validate_config = model_validator(mode='before')(
        Processor.validate_processor_before)

    @model_validator(mode='after')
    def validate_foxdenprovenancereader_after(self):
        """Validate the model configuration.

        :return: Validated model configuration
        :rtype: dict
        """
        if self.get_schema() is None:
            self.schema_ = 'foxden.reader.FoxdenProvenanceReader'
        assert self.config.url is not None
        return self

    def read(self):
        """Read records from the
        `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
        Provenance service based on did or an arbitrary query.

        :return: Provenance input and output file records.
        :rtype: list
        """
        # FIX right now the provenance reader only returns input and
        # output files, not the did. So you always also have to read a
        # metadata record
        self.logger.debug(f'config: {self.config}')

        # Submit HTTP request and return response
        rurl = f'{self.config.url}/files?did={self.config.did}'
        payload = self.config.create_http_request_payload(self)
        self.logger.info(f'method=GET url={rurl} payload={payload}')
        response = HTTP_request(rurl, payload, method='GET', scope='read')
        if self.config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = [{'name': v['name'], 'file_type': v['file_type']}
                      for v in json.loads(response.text)]
            self.logger.debug(f'Returning {len(result)} records')
            return result
        self.logger.warning(f'HTTP error code {response.status_code}')
        self.logger.warning(
            f'HTTP response:\n{response.__dict__}\npayload:\n{payload}')
        return []


class FoxdenSpecScansReader(PipelineItem):
    """Reader for `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
    SpecScans data from a specific FOXDEN SpecScans service.
    """
    def read(
            self, url, data, *, did='', query='', spec=None, method='POST',
            # 'GET',
            verbose=False):
        """Read and return data from a specific
        `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
        SpecScans service.

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (DID).
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
        # TODO FIX
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
        response = HTTP_request(rurl, payload, method=method)
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            data = json.loads(response.text)
        else:
            data = []
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.reader import main

    main()
