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

        #print(f'\nProcessor before')
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
                        #print(f'\n--> trying to get "{k}" for "{mmc}" from pipeline for schema "{schema}"')
#                        print(f'from data {type(data["data"])}')
#                        pprint(data['data'])
                        try:
#                            print('try getting value')
                            value = deepcopy(mmc.get_data(
                                data['data'], schema=schema, remove=False))
                            #print(f'Got a value {type(value)}')
#                            pprint(value)
#                            print()
                        except:
                            #print(f'Unable to get "{k}" for {mmc}')
                            pass
                        else:
#                            print(f'try validating value with merge_key_paths: {merge_key_paths}')
                            if k in data:
                                #print(f'Updating data[{k}]')
#                                pprint(data[k])
#                                print()
                                data[k] = dictionary_update(
                                    value, data[k],
                                    merge_key_paths=merge_key_paths,
                                    sort=True)
#                                print(f'-> value updated to')
#                                pprint(data[k])
#                                print()
                            else:
                                #print(f'Setting value for {k}')
                                data[k] = value
#                                print(f'data[{k}] set to')
#                                pprint(data[k])
#                                print()
        return data

#    @model_validator(mode='after')
#    def validate_processor_after(self):
#        print(f'\nProcessor after {type(self)}:')
#        pprint(self.model_dump())
#        print('\n')
#        return self

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
