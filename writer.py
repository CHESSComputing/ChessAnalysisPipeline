"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module
"""

# system modules

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
        self.__name__ = "Writer"

    def write(self, data, fileName):
        """
        write API

        :param fileName: input file name
        :return: specific number of bytes from a file
        """
        with open(fileName, 'a') as file:
            file.write(data)
        return data
