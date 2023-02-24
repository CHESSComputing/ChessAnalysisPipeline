#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module
"""

# system modules
import argparse
import json
from nexusformat.nexus import NXdata, NXfield, NXobject
import os
import sys
import xarray as xr
import yaml

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

    def write(self, data, filename):
        """
        write API

        :param filename: Name of file to write to
        :param data: data to write to file
        :return: data written to file
        """
        with open(filename, 'a') as file:
            file.write(data)
        return(data)

class YAMLWriter(Writer):
    def write(self, data, filename, force_overwrite=False):
        '''If `data` is a `dict`, write it to `filename`.

        :param data: the dictionary to write to `filename`.
        :type data: dict
        :param filename: name of the file to write to.
        :type filename: str
        :param force_overwrite: flag to allow data in `filename` to be
            overwritten if it already exists.
        :type force_overwrite: bool
        :raises TypeError: if `data` is not a `dict`
        :raises RuntimeError: if `filename` already exists and
            `force_overwrite` is `False`.
        :return: the original input data
        :rtype: dict
        '''
        print(f'{self.__name__}: write YAML data to {filename}')

        if not isinstance(data, (dict, list)):
            raise(TypeError(f'{self.__name__}.write: input data must be a dict or list.'))

        if not force_overwrite:
            if os.path.isfile(filename):
                raise(RuntimeError(f'{self.__name__}: {filename} already exists.'))

        with open(filename, 'w') as outf:
            yaml.dump(data, outf, sort_keys=False)

        return(data)

class NexusWriter(Writer):
    def write(self, data, filename, force_overwrite=False):
        '''Write `data` to a NeXus file

        :param data: the data to write to `filename`.
        :param filename: name of the file to write to.
        :param force_overwrite: flag to allow data in `filename` to be
            overwritten, if it already exists. 
        :return: the original input data
        '''
        
        print(f'{self.__name__}: write NeXus data to {filename}')
        
        if isinstance(data, NXobject):
            nxstructure = data

        elif isinstance(data, xr.Dataset):
            nxstructure = self.get_nxdata_from_dataset(data)

        elif isinstance(data, xr.DataArray):
            nxstructure = self.get_nxdata_from_dataarray(data)
        
        else:
            raise(TypeError(f'{self.__name__}.write: unknown data format: {type(data).__name__}'))

        mode = 'w' if force_overwrite else 'w-'
        nxstructure.save(filename, mode=mode)

        return(data)


    def get_nxdata_from_dataset(self, dset):
        '''Return an instance of `nexusformat.nexus.NXdata` that represents the
        data and metadata attributes contained in `dset`.

        :param dset: the input dataset to represent
        :type data: xarray.Dataset
        :return: `dset` represented as an instance of `nexusformat.nexus.NXdata`
        :rtype: nexusformat.nexus.NXdata
        '''

        nxdata_args = {'signal':None, 'axes':()}

        for var in dset.data_vars:
            data_var = dset[var]
            nxfield = NXfield(data_var.data,
                              name=data_var.name,
                              attrs=data_var.attrs)
            if nxdata_args['signal'] is None:
                nxdata_args['signal'] = nxfield
            else:
                nxdata_args[var] = nxfield

        for coord in dset.coords:
            coord_var = dset[coord]
            nxfield = NXfield(coord_var.data,
                              name=coord_var.name,
                              attrs=coord_var.attrs)
            nxdata_args['axes'] = (*nxdata_args['axes'], nxfield)

        nxdata = NXdata(**nxdata_args)
        nxdata.attrs['xarray_attrs'] = json.dumps(dset.attrs)

        return(nxdata)

    def get_nxdata_from_dataarray(self, darr):
        '''Return an instance of `nexusformat.nexus.NXdata` that represents the
        data and metadata attributes contained in `darr`.

        :param darr: the input dataset to represent
        :type darr: xarray.DataArray
        :return: `darr` represented as an instance of `nexusformat.nexus.NXdata`
        :rtype: nexusformat.nexus.NXdata
        '''
        nxdata_args = {'signal':None, 'axes':()}

        nxdata_args['signal'] = NXfield(darr.data,
                                        name=darr.name,
                                        attrs=darr.attrs)


        for coord in darr.coords:
            coord_var = darr[coord]
            nxfield = NXfield(coord_var.data,
                              name=coord_var.name,
                              attrs=coord_var.attrs)
            nxdata_args['axes'] = (*nxdata_args['axes'], nxfield)

        nxdata = NXdata(**nxdata_args)
        nxdata.attrs['xarray_attrs'] = json.dumps(darr.attrs)

        return(nxdata)


class OptionParser():
    '''User based option parser'''
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--data", action="store",
            dest="data", default="", help="Input data")
        self.parser.add_argument("--filename", action="store",
            dest="filename", default="", help="Output file")
        self.parser.add_argument("--writer", action="store",
            dest="writer", default="Writer", help="Writer class name")

def main():
    '''Main function'''
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    clsName = opts.writer
    try:
        writerCls = getattr(sys.modules[__name__],clsName)
    except:
        print(f'Unsupported writer {clsName}')
        sys.exit(1)

    writer = writerCls()
    data = writer.write(opts.data, opts.filename)
    print(f"Writer {writer} writes to {opts.filename}, data {data}")

if __name__ == '__main__':
    main()
