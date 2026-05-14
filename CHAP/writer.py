#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module defining the base `Writer` class to derive all others from.
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
from CHAP.pipeline import (
    PipelineData,
    PipelineItem,
)


def validate_writer_model(writer):
    """Validate the writer configuration.

    :return: Validated model.
    :rtype: Any
    """
    writer.filename = os.path.normpath(os.path.realpath(
        os.path.join(writer.outputdir, writer.filename)))
    if (not writer.force_overwrite
            and os.path.isfile(writer.filename)):
        raise ValueError(
            'Writing to an existing file without overwrite permission. '
            f'permission. Remove {writer.filename} or set '
            '"force_overwrite" in pipeline configuration for '
            f'{writer.name}')
    return writer


class Writer(PipelineItem):
    """Base writer.

    The job of any `Writer` in a pipeline is to receive input returned
    as a previous `PipelineItem` and write its data to file in a
    particular file format.

    :ivar filename: Name of file to write to.
    :vartype filename: str
    :ivar force_overwrite: Flag to allow data in `filename` to be
        overwritten if it already exists, defaults to `False`.
    :vartype force_overwrite: bool, optional
    :ivar remove: Flag to remove the dictionary from `data`,
        defaults to `False`.
    :vartype remove: bool, optional
    """

    filename: str
    force_overwrite: Optional[bool] = False
    remove: Optional[bool] = False

    _validate_filename = model_validator(mode='after')(validate_writer_model)

    def _update_provenance(self, data):
        """Add output file name to the provenance.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: Provenance with added output file name.
        :rtype: PipelineData
        """
        # System modules
        from os import path as os_path

        try:
            provenance = self.get_data(
                data, schema='foxden.reader.FoxdenProvenanceReader')
        except ValueError:
            return None
        output_files = provenance.pop('output_files', [])
        output_files.append({
            'name': os_path.realpath(self.filename)})
        provenance['output_files'] = output_files
        return PipelineData(
            name=self.__name__, data=provenance,
            schema='foxden.reader.FoxdenProvenanceReader')

    def write(self, data):
        """Write the last `PipelineData` item in `data` as text to a
        file.

        :param data: Input data.
        :type data: list[PipelineData]
        """
        data = self.get_pipelinedata_item(data, remove=self.remove)
        if os.path.isfile(self.filename) and not self.force_overwrite:
            raise FileExistsError(f'{self.filename} already exists')
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(data)
        self.status = 'written' # Right now does nothing yet, but could
                                # add a sort of modification flag later


class _OptionParser():
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


def main(opt_parser=_OptionParser):
    """Main function.

    :param opt_parser: User based option parser.
    :type opt_parser: CHAP.writer._OptionParser
    """
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
