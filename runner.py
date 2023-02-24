#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : runner.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
"""

# system modules
import os
import sys
import yaml
import argparse
import importlib

# local modules
from pipeline import Pipeline


class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--config", action="store",
            dest="config", default="", help="Input configuration file")
        self.parser.add_argument("--verbose", action="store_true",
            dest="verbose", default=False, help="verbose output")


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
    config = {}
    with open(opts.config) as file:
        config = yaml.safe_load(file)
    if opts.verbose:
        print(f'config:\n{yaml.dump(config)}')
    pipeline_config = config.get('pipeline', [])
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
        module = __import__(modName)
        obj = getattr(module, clsName)()
        if opts.verbose:
            print(f"loaded {obj}")
        objects.append(obj)
        kwds.append(kwargs)
    pipeline = Pipeline(objects, kwds)
    pipeline.execute(verbose=opts.verbose)


if __name__ == '__main__':
    main()
