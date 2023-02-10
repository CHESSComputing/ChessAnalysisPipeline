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
    print("opts", opts, type(opts))
    config = {}
    with open(opts.config) as file:
        config = yaml.safe_load(file)
    print(f"config {config}")
    # config {'pipeline': ['reader.Reader', 'processor.Processor', 'fitter.Fitter', 'processor.Processor', 'writer.Writer', 'fitter.Fitter', 'writer.Writer'], 'reader.Reader': {'fileName': 'config.yaml'}}
    pipeline_config = config.get('pipeline', [])
    objects = []
    for item in pipeline_config:
        # load individual object with given name from its module
        name = list(item.keys())[0]
        modName, clsName = name.split('.')
        module = __import__(modName)
        kwargs = item[name]
        obj = getattr(module, clsName)(**kwargs)
        print(f"loaded {obj} from {name} type={type(obj)}")
        objects.append(obj)
    pipeline = Pipeline(objects)
    pipeline.execute()


if __name__ == '__main__':
    main()
