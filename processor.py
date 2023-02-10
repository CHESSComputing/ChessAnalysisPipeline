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
from basemodel import BaseModel

class Processor():
    """
    Processor represent generic processor
    """
    def __init__(self, **kwargs):
        """
        Processor constructor
        """
        self.__name__ = "Processor"
        for k,v in kwargs.items():
            setattr(self, k, v)

    def process(self, data):
        """
        process data API
        """
        # process operation is a simple print function
        data += "process part\n"
        # and we return data back to pipeline
        return data

class MapProcessor(Processor):
    def process(self, data):
        print('MapProcessor.process: fill in a data with raw detector intensities.')
        print(f'\ttitle = {self.title}')
        data = np.random.rand(5,10)
        return(data)

class IntegrationProcessor(Processor):
    def process(self, data):
        print('IntegrationConfig.process: reduce the dimensionality of data.')
        print(f'\tponi_file = {self.poni_file}')
        print(f'\tmask_file = {self.mask_file}')
        print(f'\tradial_unit = {self.radial_unit}')
        data = data.sum(axis=0)
        return(data)

class CorrectionProcessor(Processor):
    def process(self, data):
        print('CorrectionProcessor: perform linear transformation on data.')
        print(f'\tcorrection_type = {self.correction_type}')
        data = 0.1*data + 0.1
        return(data)
