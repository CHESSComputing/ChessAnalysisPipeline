#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module

Define a generic `Reader` object.
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
    """Generic file reader.

    The job of any `Reader` in a `Pipeline` is to provide data stored
    in a file to the next `PipelineItem`. Note that a `Reader` used on
    its own disrupts the flow of data in a `Pipeline` -- it does not
    receive or pass along any data returned by the previous
    `PipelineItem`.
    """

    def read(self, filename):
        """Read and return the contents of `filename` as text

        :param filename: Name of file to read from
        :type filename: str
        :return: entire contents of the file
        :rtype: str
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
