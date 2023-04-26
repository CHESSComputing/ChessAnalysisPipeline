#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module
"""

# system modules
import argparse
from inspect import getfullargspec
import logging
from sys import modules
from time import time

# local modules
from CHAP.pipeline import PipelineItem


class Reader(PipelineItem):
    """Reader represent generic file writer"""

    def read(self, filename):
        """Read and return the data from requested from `filename`

        :param filename: Name of file to read from
        :return: specific number of bytes from a file
        """

        if not filename:
            self.logger.warning(
                'No file name is given, will skip read operation')
            return None

        with open(filename) as file:
            data = file.read()
        return data


class OptionParser():
    """User based option parser"""
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
    """Main function"""

    optmgr = opt_parser()
    opts = optmgr.parser.parse_args()
    cls_name = opts.reader
    try:
        reader_cls = getattr(modules[__name__], cls_name)
    except AttributeError:
        print(f'Unsupported reader {cls_name}')
        raise

    reader = reader_cls()
    reader.logger.setLevel(getattr(logging, opts.log_level))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        '{name:20}: {message}', style='{'))
    reader.logger.addHandler(log_handler)
    data = reader.read(filename=opts.filename)

    print(f'Reader {reader} reads from {opts.filename}, data {data}')


if __name__ == '__main__':
    main()
