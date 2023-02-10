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
    def __init__(self, items=None):
        """
        Pipeline class constructor
        
        :param items: list of objects
        """
        self.items = items

    def execute(self):
        """
        execute API
        """
        data = None
        for item in self.items:
            print(f"execute {item} of name: {item.__name__}")
            if hasattr(item, 'read'):
                print(f"### call item.read from {item}")
                data = item.read()
            if hasattr(item, 'process'):
                print(f"### call item.process from {item} with data={data}")
                data = item.process(data)
            if hasattr(item, 'fit'):
                print(f"### call item.fit from {item} with data={data}")
                data = item.fit(data)
            if hasattr(item, 'write'):
                print(f"### call item.write from {item} with data={data}")
                data = item.write(data)


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
        self.fitter = self.fitter

    def read(self, fileName):
        """
        read object API
        """
        return self.reader.read(fileName)

    def write(self, data, fileName):
        """
        write object API
        """
        return self.writer.write(data, fileName)

    def process(self, data):
        """
        process object API
        """
        return self.processor.process(data)

    def fit(self, data):
        """
        fit object API
        """
        return self.fitter.fit(data)
