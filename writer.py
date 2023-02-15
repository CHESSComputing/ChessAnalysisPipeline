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

    def __init__(self):
        """
        Constructor of Writer class
        """
        self.__name__ = self.__class__.__name__

    def write(self, data):
        """
        write API

        :param filename: Name of file to write to
        :param data: data to write to file
        :return: data written to file
        """
        with open(filename, 'a') as file:
            file.write(data)
        return(data)

class NumpyWriter(Writer):
    def write(self, data, filename):
        print(f'Write data to {filename}')
        np.savetxt(filename, data)
        return(data)


class YAMLWriter(Writer):
    def write(self, data, filename):
        print(f'Write data to {filename}')
        return(data)

class NexusWriter(Writer):
    def write(self, data, filename):
        print(f'Write data to {filename}')
        return(data)
