"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module
"""

# system modules
import sys
import yaml

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

class MultipleReader(Reader):
    def read(self, readers):
        '''Return resuts from multiple `Reader`s.

        :param readers: a list where each item is a tuple representing (in order)
            the name of the reader, the specific type of the `Reader`, and a
            dictionary of arguments to pass as keywords to that `Reader`'s `read`
            method (usually: `{"filename": "<filename>"}`).
        :type readers: list[tuple[str,Reader,dict[str,str]]]
        :return: The results of calling `Reader.read(**kwargs)` for each item in `readers`.
        :rtype: list[tuple(str,object)]
        '''
        data = [(r[0], getattr(sys.modules[__name__],r[1])().read(**r[2])) for r in readers]
        return(data)

class YAMLReader(Reader):
    def read(self, filename):
        print(f'read from {filename} & return data.')
        with open(filename) as file:
            data = yaml.safe_load(file)
        return(data)

class NexusReader(Reader):
    def read(self, filename):
        print(f'read from {filename} & return data.')
