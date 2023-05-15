#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module

Define a generic `Writer` object.
"""

# system modules
import argparse
from inspect import getfullargspec
import logging
from sys import modules
from time import time

# local modules
from CHAP.pipeline import PipelineItem


class Writer(PipelineItem):
    """Generic file writer

    The job of any `Writer` in a `Pipeline` is to receive input
    returned by a previous `PipelineItem`, write its data to a
    particular file format, then return the same data unaltered so it
    can be used by a successive `PipelineItem`.
    """

    def write(self, data, filename):
        """Write the input data as text to a file.

        :param data: input data
        :type data: list[PipelineData]
        :param filename: Name of file to write to
        :type filename: str
        :return: contents of the input data
        :rtype: object
        """

        data = self.unwrap_pipelinedata(data)
        with open(filename, 'a') as file:
            file.write(data)
        return data


class OptionParser():
    """User based option parser"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument(
            '--data', action='store',
            dest='data', default='', help='Input data')
        self.parser.add_argument(
            '--filename', action='store',
            dest='filename', default='', help='Output file')
        self.parser.add_argument(
            '--writer', action='store',
            dest='writer', default='Writer', help='Writer class name')
        self.parser.add_argument(
            '--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')


def main(opt_parser=OptionParser):
    """Main function"""

    optmgr = opt_parser()
    opts = optmgr.parser.parse_args()
    cls_name = opts.writer
    try:
        writer_cls = getattr(modules[__name__], cls_name)
    except AttributeError:
        print(f'Unsupported writer {cls_name}')
        raise

    writer = writer_cls()
    writer.logger.setLevel(getattr(logging, opts.log_level))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        '{name:20}: {message}', style='{'))
    writer.logger.addHandler(log_handler)
    data = writer.write(opts.data, opts.filename)
    print(f'Writer {writer} writes to {opts.filename}, data {data}')


if __name__ == '__main__':
    main()
