#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module
"""

# system modules
import argparse
from inspect import getfullargspec
import logging
from sys import modules
from time import time


class Writer():
    """Writer represent generic file writer"""

    def __init__(self):
        """Constructor of Writer class"""
        self.__name__ = self.__class__.__name__
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def write(self, data, filename, **_write_kwargs):
        """write API

        :param filename: Name of file to write to
        :param data: data to write to file
        :return: data written to file
        """

        t0 = time()
        self.logger.info(f'Executing "write" with filename={filename}, '
                         f'type(data)={type(data)}, kwargs={_write_kwargs}')

        _valid_write_args = {}
        allowed_args = getfullargspec(self._write).args \
            + getfullargspec(self._write).kwonlyargs
        for k, v in _write_kwargs.items():
            if k in allowed_args:
                _valid_write_args[k] = v
            else:
                self.logger.warning(f'Ignoring invalid arg to _write: {k}')

        data = self._write(data, filename, **_valid_write_args)

        self.logger.info(f'Finished "write" in {time()-t0:.3f} seconds\n')

        return data

    def _write(self, data, filename):
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
