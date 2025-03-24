#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: FOXDEN writers
"""

# system modules
from time import time
import json

# CHAP modules
from CHAP.foxden.utils import HttpRequest
from CHAP.pipeline import PipelineItem


class FoxdenMetadataWriter(PipelineItem):
    """FOXDEN writer writes data to Metadata FOXDEN service."""
    def write(
            self, url, data, method='POST', headers=None, verbose=False):
        """Write data to FOXDEN Provenance service

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param method: HTTP method to use, `"POST"` for creation and
            `"PUT"` for update, defaults to `"POST"`.
        :type method: str, optional
        :param headers: HTTP headers to use.
        :type headers: dictionary, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: HTTP response from FOXDEN provenance service
        :rtype: list with dictionary entry
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data}')
        # TODO: it would be useful to perform validation of data
        if isinstance(data, list) and len(data) == 1:
            data = data[0]['data'][0]
        if not isinstance(data, dict):
            raise Exception(f'Passed data={data} is not dictionary')
        mrec = {"Schema": "Analysis", "Record": data}
        payload = json.dumps(mrec)
        if verbose:
            self.logger.info(f"method=POST url={url} payload={payload}")
        response = HttpRequest(url, payload, method='POST', scope='write')
        if verbose:
            self.logger.info(f"code={response.status_code} data={response.text}")
        data = [{'code': response.status_code, 'data': response.text}]
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data

class FoxdenProvenanceWriter(PipelineItem):
    """FOXDEN writer writes data to provenance FOXDEN service."""
    def write(
            self, url, data, method='POST', headers=None, verbose=False):
        """Write data to FOXDEN Provenance service

        :param data: Input data.
        :type data: list[PipelineData]
        :param url: URL of service.
        :type url: str
        :param method: HTTP method to use, `"POST"` for creation and
            `"PUT"` for update, defaults to `"POST"`.
        :type method: str, optional
        :param headers: HTTP headers to use.
        :type headers: dictionary, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: HTTP response from FOXDEN provenance service
        :rtype: list with dictionary entry
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with url={url} data={data}')
        rurl = f'{url}/dataset'
        # TODO: it would be useful to perform validation of data
        if isinstance(data, list) and len(data) == 1:
            data = data[0]['data'][0]
        if not isinstance(data, dict):
            raise Exception(f'Passed data={data} is not dictionary')
        payload = json.dumps(data)
        if verbose:
            self.logger.info(f"method=POST url={rurl} payload={payload}")
        response = HttpRequest(rurl, payload, method='POST', scope='write')
        if verbose:
            self.logger.info(f"code={response.status_code} data={response.text}")
        data = [{'code': response.status_code, 'data': response.text}]
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
