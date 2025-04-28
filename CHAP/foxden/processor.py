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
from CHAP.common.utils import osinfo, environments
from CHAP.processor import Processor


def inputFiles():
    """Helper function to provide input files for FOXDEN."""
    return [{'name':'/tmp/file1.png'}, {'name': '/tmp/file2.png'}]

def outputFiles():
    """Helper function to provide output files for FOXDEN."""
    return [{'name':'/tmp/file1.png'}]


class FoxdenMetadataProcessor(Processor):
    """Processor to communicate with FOXDEN Metadata server."""
    def process(self, data, suffix='analysis=CHAP', verbose=False):
        """FOXDEN Metadata server communication processor.

        :param data: Input data.
        :type data: list[PipelineData]
        :param suffix: did suffix, defaults to `'analysis=CHAP'`.
        :type suffix: string, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: Data from FOXDEN Metadata service.
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with data={data}')
        output = []
        for item in data:
            # Each item in data list is a CHAP record
            # {'name': ..., 'data': {}}
            # Get the data part of processing item
            for rec in item['data']:
                if 'did' not in rec:
                    raise KeyError('Missing did in input data record')
                if '/analysis=' in rec['did']:
                    # Strip it if it is the last part of did
                    arr = rec['did'].split('/')
                    if '/analysis=' in arr[-1]:
                        did = '/'.join(arr[:-1]) + '/' + suffix
                else:
                    did = rec['did'] + '/' + suffix
                # Construct analysis record
                rec = {'did': did, 'application': 'CHAP'}
                output.append(rec)
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return output

class FoxdenProvenanceProcessor(Processor):
    """Processor to communicate with FOXDEN provenance server."""
    def process(self, data, suffix='analysis=CHAP', verbose=False):
        """FOXDEN Provenance server communication processor.

        :param data: Input data.
        :type data: list[PipelineData]
        :param suffix: did suffix, defaults to 'analysis=CHAP'.
        :type suffix: string, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: Data from FOXDEN provenance service.
        """
        t0 = time()
        self.logger.info(
            f'Executing "process" with data={data}')
        output = []
        for item in data:
            # Each item in data list is a CHAP record
            # {'name': ..., 'data': {}}
            # Get the data part of processing item
            for rec in item['data']:
                if 'did' not in rec:
                    raise KeyError('Missing did in input data record')
                rec['did'] = rec['did'] + '/' + suffix
                rec['parent_did'] = rec['did'] 
                rec['scripts'] = [
                    {'name': 'CHAP', 'parent_script': None, 'order_idx': 1}]
                rec['site'] = 'Cornell'
                rec['osinfo'] = osinfo()
                rec['environments'] = environments()
                rec['input_files'] = inputFiles()
                rec['output_files'] = outputFiles()
                rec['processing'] = 'CHAP pipeline'
                output.append(rec)
        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')
        return output


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
