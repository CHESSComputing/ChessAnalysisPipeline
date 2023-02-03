#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : fitter.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Fitter module
"""

# system modules

# local modules
# from pipeline import PipelineObject


class Fitter():
    """
    Fitter represent generic fitter
    """
    def __init__(self, func=None):
        """
        Fitter constructor
        """
        self.__name__ = "Fitter"
        self.func = func

    def fit(self, data):
        """
        fit API
        """
        # fit operation
        # for demonstratio we'll use print function
        data += "fitted part\n"
        if self.func:
            self.func(data)
        else:
            print(data)
        return data
