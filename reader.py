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
        self.__name__ = "Reader"

    def read(self, fileName):
        """
        read API

        :param fileName: input file name
        :return: specific number of bytes from a file
        """
        if not fileName:
            print(f"{__name__} no file name is given, will skip read operation")
            return None

        with open(fileName) as file:
            data = file.read()
        return data
