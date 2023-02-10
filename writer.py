"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module
"""

# system modules
import numpy as np

# local modules
# from pipeline import PipelineObject

class Writer():
    """
    Writer represent generic file writer
    """

    def __init__(self, **kwargs):
        """
        Constructor of Writer class
        """
        self.__name__ = "Writer"
        for k,v in kwargs.items():
            setattr(self, k, v)

    def write(self, data):
        """
        write API

        :param data: data to write to file
        :return: data written to file
        """
        with open(self.fileName, 'a') as file:
            file.write(data)
        return(data)

class NumpyWriter(Writer):
    def write(self, data):
        print(f'Write data to {self.fileName}')
        np.savetxt(self.fileName, data)
        return(data)


class YAMLWriter(Writer):
    def write(self, data, fileName):
        print(f'Write data to {self.fileName}')
        return(data)

class NexusWriter(Writer):
    def write(self, data, fileName):
        print(f'Write data to {self.fileName}')
        return(data)
