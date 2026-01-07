#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module

Define a generic `Reader` object.
"""

# System modules
import argparse
import logging
import os
from sys import modules

# Third party modules
from pydantic import (
    PrivateAttr,
    constr,
    model_validator,
)

# Local modules
from CHAP.pipeline import PipelineItem


def validate_reader_model(reader):
    reader._mapping_filename = reader.filename
    filename = os.path.normpath(os.path.realpath(
        os.path.join(reader.inputdir, reader.filename)))
    if (not os.path.isfile(filename)
            and not os.path.dirname(reader.filename)):
        reader.logger.warning(
            f'Unable to find {reader.filename} in '
            f'{reader.inputdir}, looking in '
            f'{reader.outputdir}')
        filename = os.path.normpath(os.path.realpath(
            os.path.join(reader.outputdir, reader.filename)))
    # Note that reader.filename has str type instead of FilePath
    # since its existence is not yet gueranteed (it can be writen
    # over the course of the pipeline's execution). So postpone
    # validation until the entire pipeline gets validated.
    if not os.path.isfile(filename):
        reader.logger.warning(
            f'Unable to find {reader.filename} during validation')
    reader.filename = filename
    return reader


class Reader(PipelineItem):
    """Generic file reader.

    The job of any `Reader` in a `Pipeline` is to provide data stored
    in a file to the next `PipelineItem`. Note that a `Reader` used on
    its own disrupts the flow of data in a `Pipeline` -- it does not
    receive or pass along any data returned by the previous
    `PipelineItem`.

    :ivar filename: Name of file to read from.
    :type filename: str
    """
    filename: constr(strip_whitespace=True, min_length=1)

    _mapping_filename: PrivateAttr(default=None)

    _validate_filename = model_validator(mode="after")(
        validate_reader_model)

    def read(self):
        """Read and return the contents of `filename` as text.

        :return: The file content.
        :rtype: str
        """
        if not self.filename:
            self.logger.warning(
                'No file name is given, skipping read operation')
            return None
        try:
            with open(self.filename) as f:
                data = f.read()
        except Exception:
            return None
        return data


class OptionParser():
    """User based option parser."""
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
    """Main function."""
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
