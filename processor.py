#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module
"""

# system modules
import argparse
import numpy as np
import sys
import xarray as xr

# local modules
# from pipeline import PipelineObject
from map import MapConfig
from integration import IntegrationConfig

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
        '''Process a map configuration & return an `xarray.Dataset` of the proper
        shape. If any scalar valued datasets are included in the map
        configuration, their values over the map will be included as `data_vars`
        in the returned `xarray.Dataset`.
        
        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :return: Map data & metadata (SPEC only, no detector)
        :rtype: xarray.Dataset
        '''

        print('MapProcessor.process: construct xr.Dataset with proper shape & metadata.')

        map_config = self.get_map_config(data)
        if not isinstance(map_config, MapConfig):
            raise(ValueError(f'{self.__name__}.process: input data is not a valid map configuration.'))

        processed_data = xr.Dataset(data_vars=self.get_data_vars(map_config),
                                    coords=self.get_coords(map_config),
                                    attrs=map_config.dict())

        return(processed_data)

    def get_map_config(self, data):
        '''Get an instance of `MapConfig` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid `MapConfig` cannot be constructed from `data`.
        :return: a valid instance of `MapConfig` with field values taken from `data`.
        :rtype: MapConfig
        '''

        print(f'{self.__name__}: get MapConfig')

        map_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'MapConfig':
                        map_config = item.get('data')
                        break

        if not map_config:
            raise(ValueError('No map configuration found'))

        return(MapConfig(**map_config))
        
    def get_coords(self, map_config):
        '''Get a dictionary of the coordinates at which the map configuration
        collected data.

        :param map_config: a valid map configuraion
        :type map_confg: MapConfig
        :return: a dictionary of coordinate names & values over the map
        :rtype: dict[str,np.ndarray]'''

        print(f'{self.__name__}: get coords dict')

        coords = {}
        for dim in map_config.independent_dimensions[::-1]:
            coords[dim.label] = (dim.label, map_config.coords[dim.label], dict(dim))

        return(coords)

    def get_data_vars(self, map_config):
        '''Get a dictionary of the scalar-valued data specified in `map_config`.

        :param map_confg: a valid map configuration
        :return: a dictionary of data labels & their values over the map
        :rtype: dict[str,np.ndarray]'''

        print(f'{self.__name__}: get data_vars dict')
        
        data_vars = {data.label: (map_config.dims, 
                                  np.empty(map_config.shape),
                                  data.dict()) for data in map_config.all_scalar_data}
        
        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(scan_number, scan_step_index, map_config)
                    for data in map_config.all_scalar_data:
                        data_vars[data.label][1][map_index] = data.get_value(scans, scan_number, scan_step_index)

        return(data_vars)


class IntegrationProcessor(Processor):
    '''Class representing a process that takes a map of 2D detector data and
    generates a map of integrated data.'''

    def process(self, data):
        '''Process an integration configuration & return a map of integrated
        data.

        :param data: input map & integration configurations, as returned from
            `MultipleReader.read`
        :type data: dict[typing.Literal['map_config','integration_config'],object]
        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key, and at least one item has
            the value `'IntegrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :return: map of integrated data
        :rtype: xr.DataArray
        '''

        map_config, integration_config = self.get_configs(data)

        integrated_data = xr.DataArray(data=self.get_data(map_config, integration_config),
                                       coords=self.get_coords(map_config, integration_config),
                                       attrs={'units':'Intensity (a.u)',
                                              'map_config': map_config.dict(),
                                              'integration_config': integration_config.dict()},
                                       name=integration_config.title)

        return(integrated_data)

    def get_configs(self, data):
        '''Return valid instances of `MapConfig` and `IntegrationConfig` from the
        input supplied by `MultipleReader`.

        :param data: input data
        :type data: dict[typing.Literal['map_config','integration_config'],object]
        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key, and at least one item has
            the value `'IntegrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises ValueError: if `data` cannot be parsed into map and integration configurations.
        :return: valid map and integration configuration objects.
        :rtype: tuple[MapConfig, IntegrationConfig]
        '''

        print(f'{self.__name__}: get map and integration configurations')

        map_config = False
        integration_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if schema == 'MapConfig':
                        map_config = item.get('data')
                    elif schema == 'IntegrationConfig':
                        integration_config = item.get('data')

        if not map_config:
            raise(ValueError('No map configuration found'))
        if not integration_config:
            raise(ValueError('No integration configuration found'))

        return(MapConfig(**map_config), IntegrationConfig(**integration_config))



    def get_data(self, map_config, integration_config):
        '''Get a numpy array of integrated data for the map and integration
        configurations provided.

        :param map_config: a valid map configuraion
        :type map_config: MapConfig
        :param integration_confg: a valid integration configuration
        :type integration_config: IntegrationConfig
        :return: an array of the integrated data specified by `map_config` and `integration_config`
        :rtype: np.ndarray
        '''
        
        print(f'{self.__name__}: Get map of integrated data')
        data = np.empty((*map_config.shape, *integration_config.integrated_data_shape))
        
        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(scan_number, scan_step_index, map_config)
                    data[map_index] = integration_config.get_integrated_data(scans, scan_number, scan_step_index)

        return(data)

    def get_coords(self, map_config, integration_config):
        '''Get a dictionary of coordinates for navigating the map's integrated
        detector intensities.

        :param map_config: a valid map configuraion
        :type map_config: MapConfig
        :param integration_confg: a valid integration configuration
        :type integration_config: IntegrationConfig
        :return: a dictionary of coordinate names & values over the map's
            integrated detector data
        :rtype: dict[str,np.ndarray]
        '''

        print(f'{self.__name__}: Get coordinates for map of integrated data')
        coords = {}

        for dim in map_config.independent_dimensions[::-1]:
            coords[dim.label] = (dim.label, map_config.coords[dim.label], dim.dict())
        
        for direction,values in integration_config.integrated_data_coordinates.items():
            coords[direction] = (direction, values, {'units':getattr(integration_config, f'{direction}_units')})

        return(coords)


class MCACeriaCalibrationProcessor(Processor):
    '''Class representing the procedure to use a CeO2 scan to obtain tuned values
    for the bragg diffraction angle and linear correction parameters for MCA
    channel energies for an EDD experimental setup.
    '''

    def process(self, data):
        '''Return tuned values for 2&theta and linear correction parameters for
        the MCA channel energies.

        :param data: input configuration for the raw data & tuning procedure
        :type data: dict
        :return: dictionary of tuned values
        :rtype: dict[str,float]
        '''

        print(f'{self.__name__}: tune 2theta & MCA energy correction parameters')

        calibration_config = self.get_config(data)

        calibrated_values = {'tth_calibrated': 7.55,
                             'slope_calibrated': 0.99,
                             'intercept_calibrated': 0.01}
        calibration_config['detector'].update(calibrated_values)
        return(calibration_config)

    def get_config(self, data):
        '''Get an instance of the configuration object needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item has the
            value `'MCACeriaCalibrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid config object cannot be constructed from `data`.
        :return: a valid instance of a configuration object with field values
            taken from `data`.
        :rtype: MCACeriaCalibrationConfig
        '''

        print(f'{self.__name__}: get MCACeriaCalibrationConfig')

        calibration_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'MCACeriaCalibrationConfig':
                        calibration_config = item.get('data')
                        break

        if not calibration_config:
            raise(ValueError('No MCA ceria calibration configuration found in input data'))

        return(calibration_config)


class MCADataProcessor(Processor):
    '''Class representing a process to return data from a MCA, restuctured to
    incorporate the shape & metadata associated with a map configuration to
    which the MCA data belongs, and linearly transformed according to the
    results of a ceria calibration.
    '''

    def process(self, data):
        '''Process configurations for a map and MCA detector(s), and return the
        raw MCA data collected over the map.

        :param data: input map configuration and results of ceria calibration
        :type data: dict[typing.Literal['map_config','ceria_calibration_results'],dict]
        :return: calibrated MCA data
        :rtype: xarray.Dataset
        '''

        print(f'{self.__name__}: gather MCA data into a map.')

        map_config, calibration_config = self.get_configs(data)

        return(data)

    def get_configs(self, data):
        '''Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key, and at least one item has
            the value `'MCACeriaCalibrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If valid config objects cannot be constructed from `data`.
        :return: valid instances of the configuration objects with field values
            taken from `data`.
        :rtype: tuple[MapConfig, MCACeriaCalibrationConfig]
        '''

        print(f'{self.__name__}: get MCACeriaCalibrationConfig')

        map_config = False
        calibration_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if schema == 'MapConfig':
                        map_config = item.get('data')
                    elif schema == 'MCACeriaCalibrationConfig':
                        calibration_config = item.get('data')

        if not map_config:
            raise(ValueError('No map configuration found in input data'))
        if not calibration_config:
            raise(ValueError('No MCA ceria calibration configuration found in input data'))

        return(MapConfig(**map_config), calibration_config)


class StrainAnalysisProcessor(Processor):
    '''Class representing a process to compute a map of sample strains by fitting
    bragg peaks in 1D detector data and analyzing the difference between measured
    peak locations and expected peak locations for the sample measured.
    '''

    def process(self, data):
        '''Process the input map detector data & configuration for the strain
        analysis procedure, and return a map of sample strains.

        :param data: results of `MutlipleReader.read` containing input map
            detector data and strain analysis configuration
        :type data: dict[list[str,object]]
        :return: map of sample strains
        :rtype: xarray.Dataset
        '''

        print(f'{self.__name__}: compute sample strain map')

        strain_analysis_config = self.get_config(data)

        return(data)

    def get_config(self, data):
        '''Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item has the
            value `'StrainAnalysisConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If valid config objects cannot be constructed from `data`.
        :return: valid instances of the configuration objects with field values
            taken from `data`.
        :rtype: StrainAnalysisConfig
        '''

        print(f'{self.__name__}: get StrainAnalysisConfig')

        strain_analysis_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if item.get('schema') == 'StrainAnalysisConfig':
                        strain_analysis_config = item.get('data')

        if not strain_analysis_config:
            raise(ValueError('No strain analysis configuration found in input data'))

        return(strain_analysis_config)


class OptionParser():
    '''User based option parser'''
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--data", action="store",
            dest="data", default="", help="Input data")
        self.parser.add_argument("--processor", action="store",
            dest="processor", default="Processor", help="Processor class name")

def main():
    '''Main function'''
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    clsName = opts.processor
    try:
        processorCls = getattr(sys.modules[__name__],clsName)
    except:
        print(f'Unsupported processor {clsName}')
        sys.exit(1)

    processor = processorCls()
    data = processor.process(opts.data)
    print(f"Processor {processor} operates on data {data}")

if __name__ == '__main__':
    main()
