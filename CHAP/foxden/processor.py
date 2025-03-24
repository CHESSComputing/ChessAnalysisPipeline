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
from CHAP.common import osinfo, environments

class FoxdenMetadataProcessor(Processor):
    """A Processor to communicate with FOXDEN Metadata server."""

    def process(self, data, suffix='analysis=CHAP', verbose=False):
        """FOXDEN Metadata processor

        :param data: Input data.
        :type data: list[PipelineData]
        :param suffix: did suffix to add, default 'analysis=CHAP'
        :type suffix: string, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: data from FOXDEN Metadata service
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with data={data}')
        output = []
        for item in data:
            # each item in data list is a CHAP record {'name': ..., 'data': {}}
            for rec in item['data']:  # get data part of processing item
                if 'did' not in rec:
                    raise Exception('No did found in input data record')
                did = rec['did'] + '/' + suffix
                # construct analysis record
                rec = {'did': did, 'application': 'CHAP'}
                output.append(rec)
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return output

class FoxdenProvenanceProcessor(Processor):
    """A Processor to communicate with FOXDEN provenance server."""
    def process(self, data, suffix='analysis=CHAP', verbose=False):
        """FOXDEN Provenance processor

        :param data: Input data.
        :type data: list[PipelineData]
        :param suffix: did suffix to add, default 'analysis=CHAP'
        :type suffix: string, optional
        :param verbose: verbose output
        :type verbose: bool, optional
        :return: data from FOXDEN provenance service
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with data={data}')
        output = []
        for item in data:
            # each item in data list is a CHAP record {'name': ..., 'data': {}}
            for rec in item['data']:  # get data part of processing item
                if 'did' not in rec:
                    raise Exception('No did found in input data record')
                rec['did'] = rec['did'] + '/' + suffix
                rec['parent_did'] = rec['did'] 
                rec['scripts'] = [{'name': 'CHAP', 'parent_script': None, 'order_idx': 1}]
                rec['site'] = 'Cornell'
                rec['osinfo'] = osinfo()
                rec['environments'] = environments()
                rec['input_files'] = inputFiles()
                rec['output_files'] = outputFiles()
                rec['processing'] = 'CHAP pipeline'
                output.append(rec)
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return output

def inputFiles():
    """
    Helper function to provide input files for FOXDEN
    """
    return [{'name':'/tmp/file1.png'}, {'name': '/tmp/file2.png'}]

def outputFiles():
    """
    Helper function to provide output files for FOXDEN
    """
    return [{'name':'/tmp/file1.png'}]

if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
