#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Reader module
"""

# system modules
import argparse
import json
from nexusformat.nexus import nxload, NXfield
import sys
import xarray as xr
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

        :param readers: a dictionary where the keys are specific names that are
            used by the next item in the `Pipeline`, and the values are `Reader`
            cconfigurations.
        :type readers: dict[str, dict]
        :return: The results of calling `Reader.read(**kwargs)` for each item
            configured in `readers`.
        :rtype: dict[str,object]
        '''
        data = {}
        for k,v in readers.items():
            reader_name = list(v.keys())[0]
            reader_class = getattr(sys.modules[__name__], reader_name)
            reader = reader_class()
            reader_kwargs = v[reader_name]

            data[k] = reader.read(**reader_kwargs)

        return(data)

class YAMLReader(Reader):
    def read(self, filename):
        '''Return a dictionary from the contents of a yaml file.

        :param filename: name of the YAML file to read from
        :return: the contents of `filename`
        :rtype: dict
        '''

        print(f'{self.__name__}: read from {filename} & return data.')
        with open(filename) as file:
            data = yaml.safe_load(file)
        return(data)

class NexusReader(Reader):
    def read(self, filename, nxpath='/'):
        '''Return an instance of `xarray.Dataset` representing the contents of
        the default `nexusformat.nexus.NXdata` object in `filename`.

        :param filename: name of the NeXus file to read from
        :param nxpath: path to a specific loaction in the NeXus file to read from,
            defaults to `"/"`
        :type nxpath: str, optional
        :return: the default plottable data in `filename`
        :rtype: xarray.Dataset
        '''

        print(f'{self.__name__}: read from {filename} & return data.')
        
        nxobject = nxload(filename)[nxpath]
        nxdata = nxobject.plottable_data

        data_vars = {}
        coords = {}
        for nxname,nxobject in nxdata.items():
            if isinstance(nxobject, NXfield):
                if nxname in nxdata.attrs['axes']:
                    coords[nxname] = (nxname,
                                      nxobject.nxdata,
                                      {k:v.nxvalue for k,v in nxobject.attrs.items()})
                else:
                    data_vars[nxname] = (nxdata.attrs['axes'],
                                         nxobject.nxdata,
                                         {k:v.nxvalue for k,v in nxobject.attrs.items()})
        
        if 'xarray_attrs' in nxdata.attrs:
            attrs = json.loads(nxdata.attrs['xarray_attrs'])
        else:
            attrs = {}

        dset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        return(dset)


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
