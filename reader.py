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

    def __init__(self, **kwargs):
        """
        Constructor of Reader class
        """
        self.__name__ = "Reader"
        for k,v in kwargs.items():
            setattr(self, k, v)

    def read(self):
        """
        read API

        :return: specific number of bytes from a file
        """
        if not self.fileName:
            print(f"{__name__} no file name is given, will skip read operation")
            return None

        with open(self.fileName) as file:
            data = file.read()
        return data


class YAMLReader(Reader):
    def read(self):
        print(f'read from {self.fileName} & return data.')

class NexusReader(Reader):
    def read(self):
        print(f'read from {self.fileName} & return data.')
