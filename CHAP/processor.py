#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module

Define a generic `Processor` object.
"""

# System modules
import argparse
import logging
from sys import modules
from typing import Optional

# Third party modules
from pydantic import model_validator

# Local modules
from CHAP.pipeline import PipelineItem


class Processor(PipelineItem):
    """Generic data processor.

    The job of any `Processor` in a `Pipeline` is to receive data
    returned by the previous `PipelineItem`, process it in some way,
    and return the result for the next `PipelineItem` to use as input.
    """
    @model_validator(mode='before')
    @classmethod
    def validate_processor_before(cls, data):
        # System modules
        from copy import deepcopy

        # Local modules
        from CHAP.utils.general import (
            dictionary_update,
            is_str_or_str_series,
        )

        if isinstance(data, dict):
            if 'data' in data and 'modelmetaclass' in data:
                mmc = data['modelmetaclass']
                pipeline_fields = mmc.model_fields.get('pipeline_fields')
                if pipeline_fields is not None:
                    for k, v in pipeline_fields.default.items():
                        if is_str_or_str_series(v, log=False):
                            schema = v
                            merge_key_paths = None
                        else:
                            schema = v.get('schema')
                            merge_key_paths = v.get('merge_key_paths')
                        try:
                            value = deepcopy(mmc.get_data(
                                data['data'], schema=schema, remove=False))
                        except:
                            pass
                        else:
                            if k in data:
                                data[k] = dictionary_update(
                                    value, data[k],
                                    merge_key_paths=merge_key_paths,
                                    sort=True)
                            else:
                                data[k] = value
        return data

    def process(self, data):
        """Extract the contents of the input data, add a string to it,
        and return the amended value.

        :param data: Input data.
        :return: Processed data.
        """
        # If needed, extract data from a returned value of Reader.read
        if isinstance(data, list):
            if all(isinstance(d, dict) for d in data):
                data = data[0]['data']
        if data is None:
            return []
        # The process operation is a simple string concatenation
        data += 'process part\n'
        # Return data back to pipeline
        return data


class OptionParser():
    """User based option parser."""
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
    """Main function."""
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
    processor.process(opts.data)


if __name__ == '__main__':
    main()
