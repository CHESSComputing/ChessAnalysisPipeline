#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : pipeline.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
"""

# system modules


class Pipeline():
    """
    Pipeline represent generic Pipeline class
    """
    def __init__(self, items=None, kwds=None):
        """
        Pipeline class constructor
        
        :param items: list of objects
        :param kwds: list of method args for individual objects
        """
        self.items = items
        self.kwds = kwds

    def execute(self, verbose=False):
        """
        execute API
        """
        data = None
        for item, kwargs in zip(self.items, self.kwds):
            if hasattr(item, 'read'):
                if verbose:
                    print(f"### call item.read from {item} with kwargs={kwargs}")
                data = item.read(**kwargs)
            if hasattr(item, 'process'):
                if verbose:
                    print(f"### call item.process from {item} with data={data} kwargs={kwargs}")
                data = item.process(data, **kwargs)
            if hasattr(item, 'write'):
                if verbose:
                    print(f"### call item.write from {item} with data={data} kwargs={kwargs}")
                data = item.write(data, **kwargs)


class PipelineObject():
    """
    PipelineObject represent generic Pipeline class
    """
    def __init__(self, reader, writer, processor, fitter):
        """
        PipelineObject class constructor
        """
        self.reader = reader
        self.writer = writer
        self.processor = processor

    def read(self, filename):
        """
        read object API
        """
        return self.reader.read(filename)

    def write(self, data, filename):
        """
        write object API
        """
        return self.writer.write(data, filename)

    def process(self, data):
        """
        process object API
        """
        return self.processor.process(data)

