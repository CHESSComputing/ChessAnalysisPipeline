#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : runner.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
"""

# system modules
import argparse
import logging
import os
import sys
import yaml

# local modules
from CHAP.pipeline import Pipeline


class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--config", action="store",
            dest="config", default="", help="Input configuration file")
        self.parser.add_argument('--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    runner(opts)

def runner(opts):
    """
    Main runner function

    :param opts: opts is an instance of argparse.Namespace which contains all input parameters
    """

    log_level = opts.log_level.upper()
    logger, log_handler = setLogger(log_level)
    config = {}
    with open(opts.config) as file:
        config = yaml.safe_load(file)
    logger.info(f'Input configuration: {config}\n')
    pipeline_config = config.get('pipeline', [])
    run(pipeline_config, logger, log_level, log_handler)

def setLogger(log_level="INFO"):
    """
    Helper function to set CHAP logger

    :param log_level: logger level, default INFO
    """
    logger = logging.getLogger(__name__)
    log_level = getattr(logging, log_level.upper())
    logger.setLevel(log_level)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter('{name:20}: {message}', style='{'))
    logger.addHandler(log_handler)
    return logger, log_handler

def run(pipeline_config, logger=None, log_level=None, log_handler=None):
    """
    Run given pipeline_config

    :param pipeline_config: CHAP pipeline config
    """
    objects = []
    kwds = []
    for item in pipeline_config:
        # load individual object with given name from its module
        if isinstance(item, dict):
            name = list(item.keys())[0]
            kwargs = item[name]
        else:
            name = item
            kwargs = {}
        modName, clsName = name.split('.')
        module = __import__(f'CHAP.{modName}', fromlist=[clsName])
        obj = getattr(module, clsName)()
        if log_level:
            obj.logger.setLevel(log_level)
        if log_handler:
            obj.logger.addHandler(log_handler)
        if logger:
            logger.info(f'Loaded {obj}')
        objects.append(obj)
        kwds.append(kwargs)
    pipeline = Pipeline(objects, kwds)
    if log_level:
        pipeline.logger.setLevel(log_level)
    if log_handler:
        pipeline.logger.addHandler(log_handler)
    if logger:
        logger.info(f'Loaded {pipeline} with {len(objects)} items\n')
        logger.info(f'Calling "execute" on {pipeline}')
    pipeline.execute()


if __name__ == '__main__':
    main()
