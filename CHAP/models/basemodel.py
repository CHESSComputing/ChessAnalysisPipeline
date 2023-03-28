#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : basemodel.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: BaseModel module
"""

# system modules
import logging


class BaseModel():
    """
    BaseModel docstring
    """
    def __init__(self, filename=None, **kwds):
        self.logger = logging.getLogger(__name__)
        self.construct(filename, **kwds)
        self.map = dict(name=__name__)

    def construct(self, filename=None, **kwds):
        """
        construct from CLI object

        :param filename: input file name
        :param **kwds: named arguments
        :return: Basemodel object
        """
        print('construct API calls: ', end='')
        if filename and filename.endswith('yaml'):
            self.construct_from_yaml(filename)
        elif filename and filename != '':
            self.construct_from_file(filename)
        else:
            self.construct_from_config(**kwds)

    @classmethod
    def construct_from_config(cls, **config):
        """
        construct from config object

        :param **config: named arguments
        :return: Basemodel object
        """
        print(f'construct_from_config: {config}')

    @classmethod
    def construct_from_yaml(cls, filename):
        """
        construct from CLI object

        :param filename: input file name
        :return: Basemodel object
        """
        print(f'construct_from_yaml: {filename}')

    @classmethod
    def construct_from_file(cls, filename):
        """
        construct from filename

        :param filename: input file name
        :return: Basemodel object
        """
        print(f'construct_from_file: {filename}')

    def getMap(self):
        """
        return model map

        :return: map object
        """
        return self.map


if __name__ == '__main__':
    print('### should construct from file.yaml')
    base = BaseModel('file.yaml')
    print('### should construct from file.txt')
    base = BaseModel('file.txt')
    print('### should construct from config')
    base = BaseModel(param='file.txt', arg='bla')
