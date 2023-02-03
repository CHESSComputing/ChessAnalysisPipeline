#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module
"""

# system modules

# local modules
# from pipeline import PipelineObject


class Processor():
    """
    Processor represent generic processor
    """
    def __init__(self):
        """
        Pipieline constructor
        """
        self.__name__ = "Processor"

    def process(self, data):
        """
        process data API
        """
        # process operation is a simple print function
        data += "process part\n"
        # and we return data back to pipeline
        return data
