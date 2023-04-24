#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module
"""

# system modules
import argparse
from inspect import getfullargspec
import logging
from sys import modules
from time import time


class Processor():
    """Processor represent generic processor"""
    def __init__(self):
        """Processor constructor"""
        self.__name__ = self.__class__.__name__
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def process(self, data, **_process_kwargs):
        """process data API

        :param _process_kwargs: keyword arguments to pass to
            `self._process`, defaults to `{}`
        :type _process_kwargs: dict, optional
        """

        t0 = time()
        self.logger.info(f'Executing "process" with type(data)={type(data)}')

        _valid_process_args = {}
        allowed_args = getfullargspec(self._process).args \
            + getfullargspec(self._process).kwonlyargs
        for k, v in _process_kwargs.items():
            if k in allowed_args:
                _valid_process_args[k] = v
            else:
                self.logger.warning(f'Ignoring invalid arg to _process: {k}')

        data = self._process(data, **_valid_process_args)

        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')

        return data

    def _process(self, data):
        """Private method to carry out the mechanics of the specific
        Processor.

        :param data: input data
        :return: processed data
        """
        # If needed, extract data from a returned value of Reader.read
        if isinstance(data, list):
            if all(isinstance(d, dict) for d in data):
                data = data[0]['data']
        if data is None:
            return []
        # process operation is a simple print function
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
