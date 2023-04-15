#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : pipeline.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
"""

# system modules
import logging
from time import time

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
        self.__name__ = self.__class__.__name__

        self.items = items
        self.kwds = kwds

        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def execute(self):
        """
        execute API
        """
        from CHAP.tomo.processor import TomoDataProcessor

        t0 = time()
        self.logger.info(f'Executing "execute"\n')

        data = None
        for item, kwargs in zip(self.items, self.kwds):
            if not isinstance(item, TomoDataProcessor):
                kwargs.pop('interactive')
            if hasattr(item, 'read'):
                self.logger.info(f'Calling "read" on {item}')
                data = item.read(**kwargs)
            if hasattr(item, 'process'):
                self.logger.info(f'Calling "process" on {item}')
                data = item.process(data, **kwargs)
            if hasattr(item, 'write'):
                self.logger.info(f'Calling "write" on {item}')
                data = item.write(data, **kwargs)

        self.logger.info(f'Executed "execute" in {time()-t0:.3f} seconds')

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

