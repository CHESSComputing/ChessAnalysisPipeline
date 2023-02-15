#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module
"""

# system modules
import numpy as np
import xarray as xr

# local modules
# from pipeline import PipelineObject
from map import MapConfig

class Processor():
    """
    Processor represent generic processor
    """
    def __init__(self):
        """
        Processor constructor
        """
        self.__name__ = self.__class__.__name__

    def process(self, data):
        """
        process data API
        """
        # process operation is a simple print function
        data += "process part\n"
        # and we return data back to pipeline
        return data

class MapProcessor(Processor):
    '''Class representing a process that takes a map configuration and returns a
    properly shaped `xr.Dataset` for that map. The original configuration
    metadata will be present in the `attrs` attribute. Optional scalar-valued
    data from the map will be included, if present in the supplied map
    configuration.'''

    def process(self, data):
        '''Process a map configuration & return an `xarray.Dataset` of the proper shape.
        
        :param data: Map configuration parameters
        :type data: dict
        :return: Map data & metadata (SPEC only, no detector)
        '''

        print('MapProcessor.process: construct xr.Dataset with proper shape & metadata.')

        map_config = self.get_map_config(data)
        if not isinstance(map_config, MapConfig):
            raise(ValueError(f'{self.__name__}.process: input data is not a valid map configuration.'))

        processed_data = xr.Dataset(data_vars=self.get_data_vars(map_config),
                                    coords=self.get_coords(map_config),
                                    attrs=data)

        return(processed_data)

    def get_map_config(self, data):
        '''Get an instance of `MapConfig` from an input dictionary

        :param data: input to `MapConfig`'s constructor
        :type data: dict
        :raises Exception: If a valid `MapConfig` cannot be constructed from `data`.
        :return: a valid instance of `MapConfig` with field values taken from `data`.
        :rtype: MapConfig'''

        print(f'{self.__name__}: get MapConfig from dict')
        return(MapConfig(**data))
        
    def get_coords(self, map_config):
        '''Get a dictionary of the coordinates at which the map configuration collected data.

        :param map_config: a valid map configuraion
        :return: a dictionary of coordinate names & values over the map
        :rtype: dict[str,np.ndarray]'''

        print(f'{self.__name__}: return coords dict')

        coords = {}
        for dim in map_config.independent_dimensions[::-1]:
            coords[dim.label] = (dim.label, map_config.coords[dim.label], dict(dim))

        return(coords)

    def get_data_vars(self, map_config):
        '''Get a dictionary of the scalar-valued data specified in `map_config`.

        :param map_confg: a valid map configuration
        :return: a dictionary of data labels & their values over the map
        :rtype: dict[str,np.ndarray]'''

        print(f'{self.__name__}: construct data_vars dict')
        
        data_vars = {data.label: (map_config.dims, 
                                  np.empty(map_config.shape),
                                  dict(data)) for data in map_config.all_scalar_data}
        
        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(scan_number, scan_step_index, map_config)
                    for data in map_config.all_scalar_data:
                        data_vars[data.label][1][map_index] = data.get_value(scans, scan_number, scan_step_index)

        return(data_vars)
