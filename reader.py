#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module
"""

# system modules
import argparse
import json
import sys

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

    def read(self, type_=None, schema=None, **_read_kwargs):
        '''Read API

        Wrapper to read, format, and return the data requested.

        :param type_: the expected type of data read from `filename`, defualts
            to `None`
        :type type_: type, optional
        :param schema: the expected schema of the data read from `filename`,
            defaults to `None`
        :type schema: str, otional
        :param _read_kwargs: keyword arguments to pass to `self._read`, defaults
            to `{}`
        :type _read_kwargs: dict, optional
        :return: list with one item: a dictionary containing the data read from
            `filename`, the name of this `Reader`, and the values of `type_` and
            `schema`.
        :rtype: list[dict[str,object]]
        '''

        data = [{'name': self.__name__,
                 'data': self._read(**_read_kwargs),
                 'type': type_,
                 'schema': schema}]

        return(data)

    def _read(self, filename):
        '''Read and return the data from requested from `filename`

        :param filename: Name of file to read from
        :return: specific number of bytes from a file
        '''

        if not filename:
            print(f"{self.__name__} no file name is given, will skip read operation")
            return None

        with open(filename) as file:
            data = file.read()
        return(data)

class MultipleReader(Reader):
    def read(self, readers):
        '''Return resuts from multiple `Reader`s.

        :param readers: a dictionary where the keys are specific names that are
            used by the next item in the `Pipeline`, and the values are `Reader`
            cconfigurations.
        :type readers: list[dict]
        :return: The results of calling `Reader.read(**kwargs)` for each item
            configured in `readers`.
        :rtype: list[dict[str,object]]
        '''

        print(f'{self.__name__}: read from {len(readers)} Readers')
        data = []
        for reader_config in readers:
            reader_name = list(reader_config.keys())[0]
            reader_class = getattr(sys.modules[__name__], reader_name)
            reader = reader_class()
            reader_kwargs = reader_config[reader_name]

            data.extend(reader.read(**reader_kwargs))

        return(data)

class YAMLReader(Reader):
    def _read(self, filename):
        '''Return a dictionary from the contents of a yaml file.

        :param filename: name of the YAML file to read from
        :return: the contents of `filename`
        :rtype: dict
        '''

        print(f'{self.__name__}: read dictionary from {filename}.')

        import yaml

        with open(filename) as file:
            data = yaml.safe_load(file)
        return(data)

class NexusReader(Reader):
    def _read(self, filename, nxpath='/'):
        '''Return the NeXus object stored at `nxpath` in the nexus file
        `filename`.

        :param filename: name of the NeXus file to read from
        :type filename: str
        :param nxpath: path to a specific loaction in the NeXus file to read
            from, defaults to `'/'`
        :type nxpath: str, optional
        :raises nexusformat.nexus.NeXusError: if `filename` is not a NeXus
            file or `nxpath` is not in `filename`.
        :return: the NeXus structure indicated by `filename` and `nxpath`.
        :rtype: nexusformat.nexus.NXobject
        '''

        print(f'{self.__name__}: read NeXus object at {nxpath} in {filename}')

        from nexusformat.nexus import nxload

        nxobject = nxload(filename)[nxpath]
        return(nxobject)


class OptionParser():
    '''User based option parser'''
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--filename", action="store",
            dest="filename", default="", help="Input file")
        self.parser.add_argument("--reader", action="store",
            dest="reader", default="Reader", help="Reader class name")

def main():
    '''Main function'''
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    clsName = opts.reader
    try:
        readerCls = getattr(sys.modules[__name__],clsName)
    except:
        print(f'Unsupported reader {clsName}')
        sys.exit(1)

    reader = readerCls()
    data = reader.read(opts.filename)
    print(f"Reader {reader} reads from {opts.filename}, data {data}")

if __name__ == '__main__':
    main()
