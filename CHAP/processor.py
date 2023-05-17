#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module

Define a generic `Processor` object.
"""

# system modules
import argparse
from inspect import getfullargspec
import logging
from sys import modules
from time import time

# local modules
from CHAP.pipeline import PipelineItem

class Processor(PipelineItem):
    """Generic data processor.

    The job of any `Processor` in a `Pipeline` is to receive data
    returned by the previous `PipelineItem`, process it in some way,
    and return the result for the next `PipelineItem` to use as input.
    """

    def process(self, data):
        """Extract the contents of the input data, add a string to it,
        and return the amended value.

        :param data: input data
        :return: processed data
        """
        # If needed, extract data from a returned value of Reader.read
        if isinstance(data, list):
            if all(isinstance(d, dict) for d in data):
                data = data[0]['data']
        if data is None:
            return []
        # process operation is a simple string concatenation
        data += "process part\n"
        # and we return data back to pipeline
        return data


class OptionParser():
    """User based option parser"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument(
            '--data', action='store',
            dest='data', default='', help='Input data')
        self.parser.add_argument(
            '--processor', action='store',
            dest='processor', default='Processor', help='Processor class name')
        self.parser.add_argument(
            '--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')


def main(opt_parser=OptionParser):
    """Main function"""

    optmgr = opt_parser()
    opts = optmgr.parser.parse_args()
    cls_name = opts.processor
    try:
        processor_cls = getattr(modules[__name__], cls_name)
    except AttributeError:
        print(f'Unsupported processor {cls_name}')
        raise

    processor = processor_cls()
    processor.logger.setLevel(getattr(logging, opts.log_level))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        '{name:20}: {message}', style='{'))
    processor.logger.addHandler(log_handler)
    data = processor.process(opts.data)

    print(f'Processor {processor} operates on data {data}')


if __name__ == '__main__':
    main()
