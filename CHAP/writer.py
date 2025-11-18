#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module

Define a generic `Writer` object.
"""

# System modules
import argparse
import logging
import os
from sys import modules
from typing import Optional

# Third party modules
from pydantic import model_validator

# Local modules
from CHAP.pipeline import PipelineItem


def validate_writer_model(model_instance):
    model_instance.filename = os.path.normpath(os.path.realpath(
        os.path.join(model_instance.outputdir, model_instance.filename)))
    if (not model_instance.force_overwrite
            and os.path.isfile(model_instance.filename)):
        raise ValueError(
            'Writing to an existing file without overwrite permission. '
            f'permission. Remove {model_instance.filename} or set '
            '"force_overwrite" in pipeline configuration for '
            f'{model_instance.name}')
    return model_instance


class Writer(PipelineItem):
    """Generic file writer.

    The job of any `Writer` in a `Pipeline` is to receive input
    returned by a previous `PipelineItem`, write its data to a
    particular file format, then return the same data unaltered so it
    can be used by a successive `PipelineItem`.

    :ivar filename: Name of file to write to.
    :type filename: str
    :ivar force_overwrite: Flag to allow data in `filename` to be
        overwritten if it already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    :ivar remove: Flag to remove the dictionary from `data`,
        defaults to `False`.
    :type remove: bool, optional
    """
    filename: str
    force_overwrite: Optional[bool] = False
    remove: Optional[bool] = False

    _validate_filename = model_validator(mode="after")(
        validate_writer_model)

    def write(self, data):
        """Write the last `CHAP.pipeline.PipelineData` item in `data`
        as text to a file.

        :param data: Input data.
        :type data: list[CHAP.pipeline.PipelineData]
        :return: Contents of the input data.
        :rtype: list[PipelineData]
        """
        ddata = self.unwrap_pipelinedata(data)[-1]
        if os_path.isfile(self.filename) and not self.force_overwrite:
            raise FileExistsError(f'{self.filename} already exists')
        with open(self.filename, 'w') as f:
            f.write(ddata)
        if self.remove:
            data.pop()
        self.status = 'written' # Right now does nothing yet, but could
                                # add a sort of modification flag later
        return data


class OptionParser():
    """User based option parser."""
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
    """Main function."""
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
