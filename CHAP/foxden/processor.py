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

# Local modules
from CHAP.processor import Processor

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
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
