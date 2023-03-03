#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module
"""

# system modules
import argparse
import json
import logging
import os
import sys
from time import time

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
        self.logger = logging.getLogger(self.__name__)

    def write(self, data, filename, **_write_kwargs):
        """
        write API

        :param filename: Name of file to write to
        :param data: data to write to file
        :return: data written to file
        """

        t0 = time()
        self.logger.info(f'Executing "write" with filename={filename}, data={repr(data)}, kwargs={_write_kwargs}')

        data = self._write(data, filename, **_write_kwargs)

        self.logger.info(f'Finished "write" in {time()-t0:.3f} seconds\n')

        return(data)

    def _write(self, data, filename):
        with open(filename, 'a') as file:
            file.write(data)
        return(data)

class YAMLWriter(Writer):
    def _write(self, data, filename, force_overwrite=False):
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

        import yaml

        if not isinstance(data, (dict, list)):
            raise(TypeError(f'{self.__name__}.write: input data must be a dict or list.'))

        if not force_overwrite:
            if os.path.isfile(filename):
                raise(RuntimeError(f'{self.__name__}: {filename} already exists.'))

        with open(filename, 'w') as outf:
            yaml.dump(data, outf, sort_keys=False)

        return(data)

class NexusWriter(Writer):
    def _write(self, data, filename, force_overwrite=False):
        '''Write `data` to a NeXus file

        :param data: the data to write to `filename`.
        :param filename: name of the file to write to.
        :param force_overwrite: flag to allow data in `filename` to be
            overwritten, if it already exists. 
        :return: the original input data
        '''

        from nexusformat.nexus import NXobject
        import xarray as xr
        
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

        from nexusformat.nexus import NXdata, NXfield

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

        from nexusformat.nexus import NXdata, NXfield

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
        self.parser.add_argument('--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')

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
    writer.logger.setLevel(getattr(logging, opts.log_level))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter('{name:20}: {message}', style='{'))
    writer.logger.addHandler(log_handler)
    data = writer.write(opts.data, opts.filename)
    print(f"Writer {writer} writes to {opts.filename}, data {data}")

if __name__ == '__main__':
    main()
