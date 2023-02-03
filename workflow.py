#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : workflow.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Workflow module
"""

# system modules
from basemodel import BaseModel


class Workflow(BaseModel):
    """
    Workflow docstring
    """
    def __init__(self, filename=None, **kwds):
        super().__init__(filename, **kwds)
        self.map['workflow'] = __name__
        print('create Workflow calls: ', end='')


class EDDWorkflow(Workflow):
    """
    EDDWorkflow
    """
    def __init__(self, filename=None, **kwds):
        super().__init__(filename, **kwds)
        self.map['workflow'] = 'edd'
        print('create EDDWorkflow')

class SAXWWorkflow(Workflow):
    """
    SAXWWorkflow
    """
    def __init__(self, filename=None, **kwds):
        super().__init__(filename, **kwds)
        self.map['workflow'] = 'saxw'
        print('create SAXWWorkflow')

if __name__ == '__main__':
    print('--- create EDDWorkflow from config')
    wflow = EDDWorkflow()
    print('map', wflow.map)
    print('--- create SAXWWorkflow from file.txt')
    wflow = SAXWWorkflow('file.txt')
    print('map', wflow.map)
