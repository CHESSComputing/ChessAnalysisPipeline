"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: generic Writer module
"""

# system modules
from nexusformat.nexus import NXdata, NXfield, NXobject
import xarray as xr

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

class NexusWriter(Writer):
    def write(self, data, filename):
        '''Write `data` to a NeXus file

        :param data: the data to write to `filename`.
        :param filename: name of the file to write to.
        :return: the original input data
        '''
        
        print(f'Write data to {filename}')
        
        if isinstance(data, NXobject):
            nxstructure = data

        elif isinstance(data, xr.Dataset):
            nxstructure = self.get_nxdata_from_dataset(data)

        elif isinstance(data, xr.DataArray):
            nxstructure = self.get_nxdata_from_dataarray(data)
        
        else:
            print(f'{self.__name__}.write: unknown data format {type(data)}')
            raise(TypeError(f'{self.__name__}.write: unknown data format: {type(data).__name__}'))

        nxstructure.save(filename)
        return(data)


    def get_nxdata_from_dataset(self, dset):
        '''Return an instance of `nexusformat.nexus.NXdata` that represents the
        data and metadata attributes contained in `dset`.

        :param dset: the input dataset to represent
        :type data: xarray.Dataset
        :return: `dset` represented as an instance of `nexusformat.nexus.NXdata`
        :rtype: nexusformat.nexus.NXdata
        '''
        print(f'{self.__name__}: get NXdata from xr.Dataset')

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

        return(nxdata)
