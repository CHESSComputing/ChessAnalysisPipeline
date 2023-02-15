"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module
"""

# system modules

# local modules
# from pipeline import PipelineObject

class Reader():
    """
    Reader represent generic file writer
    """

    def __init__(self):
        """
        Constructor of Reader class
        """
        self.__name__ = self.__class__.__name__

    def read(self, filename):
        """
        read API

        :param filename: Name of file to read from
        :return: specific number of bytes from a file
        """
        if not filename:
            print(f"{self.__name__} no file name is given, will skip read operation")
            return None

        with open(filename) as file:
            data = file.read()
        return data


class YAMLReader(Reader):
    def read(self, filename):
        print(f'read from {filename} & return data.')

class NexusReader(Reader):
    def read(self, filename):
        print(f'read from {filename} & return data.')
