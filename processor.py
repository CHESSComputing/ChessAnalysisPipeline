#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module
"""

# system modules
import numpy as np

# local modules
# from pipeline import PipelineObject

class Processor():
    """
    Processor represent generic processor
    """
    def __init__(self):
        """
        Processor constructor
        """
        self.__name__ = self.__class__.__name__

    def process(self, data):
        """
        process data API
        """
        # process operation is a simple print function
        data += "process part\n"
        # and we return data back to pipeline
        return data

class MapProcessor(Processor):
    '''Class representing a process that takes a map configuration and returns a
    properly shaped `xr.Dataset` for that map. The original configuration
    metadata will be present in the `attrs` attribute. Optional scalar-valued
    data from the map will be included, if present in the supplied map
    configuration.'''
    def process(self, data):
        '''Process a map configuration & return an `xarray.Dataset` of the proper shape.
        
        :param data: Map configuration parameters
        :type data: dict
        :return: Map data & metadata (SPEC only, no detector)
        '''
        print('MapProcessor.process: construct xr.Dataset with proper shape & metadata.')
        data = np.random.rand(5,10)
        return(data)

