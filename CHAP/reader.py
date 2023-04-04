#!/usr/bin/env python
'''
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module
'''

# system modules
import argparse
import json
import logging
import sys
from time import time

# local modules
# from pipeline import PipelineObject

class Reader():
    '''
    Reader represent generic file writer
    '''

    def __init__(self):
        '''
        Constructor of Reader class
        '''
        self.__name__ = self.__class__.__name__
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def read(self, type_=None, schema=None, encoding=None, **_read_kwargs):
        '''Read API

        Wrapper to read, format, and return the data requested.

        :param type_: the expected type of data read from `filename`, defualts
            to `None`
        :type type_: type, optional
        :param schema: the expected schema of the data read from `filename`,
            defaults to `None`
        :type schema: str, otional
        :param _read_kwargs: keyword arguments to pass to `self._read`, defaults
            to `{}`
        :type _read_kwargs: dict, optional
        :return: list with one item: a dictionary containing the data read from
            `filename`, the name of this `Reader`, and the values of `type_` and
            `schema`.
        :rtype: list[dict[str,object]]
        '''

        t0 = time()
        self.logger.info(f'Executing "read" with type={type_}, schema={schema}, kwargs={_read_kwargs}')

        data = [{'name': self.__name__,
                 'data': self._read(**_read_kwargs),
                 'type': type_,
                 'schema': schema,
                 'encoding': encoding}]

        self.logger.info(f'Finished "read" in {time()-t0:.3f} seconds\n')
        return(data)

    def _read(self, filename):
        '''Read and return the data from requested from `filename`

        :param filename: Name of file to read from
        :return: specific number of bytes from a file
        '''

        if not filename:
            self.logger.warning('No file name is given, will skip read operation')
            return None

        with open(filename) as file:
            data = file.read()
        return(data)

class OptionParser():
    '''User based option parser'''
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument(
            '--filename', action='store',
            dest='filename', default='', help='Input file')
        self.parser.add_argument(
            '--reader', action='store',
            dest='reader', default='Reader', help='Reader class name')
        self.parser.add_argument(
            '--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')

def main(opt_parser=OptionParser):
    '''Main function'''

    optmgr  = opt_parser()
    opts = optmgr.parser.parse_args()
    clsName = opts.reader
    try:
        readerCls = getattr(sys.modules[__name__],clsName)
    except:
        print(f'Unsupported reader {clsName}')
        sys.exit(1)

    reader = readerCls()
    reader.logger.setLevel(getattr(logging, opts.log_level))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter('{name:20}: {message}', style='{'))
    reader.logger.addHandler(log_handler)
    data = reader.read(filename=opts.filename)

    print(f'Reader {reader} reads from {opts.filename}, data {data}')

if __name__ == '__main__':
    main()
