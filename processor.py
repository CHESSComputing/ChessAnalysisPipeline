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
import json
import sys

# local modules
# from pipeline import PipelineObject

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
        # If needed, extract data from a returned value of Reader.read
        if isinstance(data, list):
            if all([isinstance(d,dict) for d in data]):
                data = data[0]['data']
        # process operation is a simple print function
        data += "process part\n"
        # and we return data back to pipeline
        return data

class MapProcessor(Processor):
    '''Class representing a process that takes a map configuration and returns a
    `nexusformat.nexus.NXentry` representing that map's metadata and any
    scalar-valued raw data requseted by the supplied map configuration.
    '''

    def process(self, data):
        '''Process the output of a `Reader` that contains a map configuration and
        return a `nexusformat.nexus.NXentry` representing the map.

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :return: Map data & metadata (SPEC only, no detector)
        :rtype: nexusformat.nexus.NXentry
        '''

        print(f'{self.__name__}: get MapConfig from input, return an NXentry')

        map_config = self.get_map_config(data)
        nxentry = self.get_nxentry(map_config)

        return(nxentry)

    def get_map_config(self, data):
        '''Get an instance of `MapConfig` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid `MapConfig` cannot be constructed from `data`.
        :return: a valid instance of `MapConfig` with field values taken from `data`.
        :rtype: MapConfig
        '''

        from models.map import MapConfig

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
        
    def get_nxentry(self, map_config):
        '''Use a `MapConfig` to construct a `nexusformat.nexus.NXentry`

        :param map_config: a valid map configuration
        :type map_config: MapConfig
        :return: the map's data and metadata contained in a NeXus structure
        :rtype: nexusformat.nexus.NXentry
        '''

        from nexusformat.nexus import (NXcollection,
                                       NXdata,
                                       NXentry,
                                       NXfield,
                                       NXsample)
        import numpy as np

        nxentry = NXentry(name=map_config.title)

        nxentry.map_config = json.dumps(map_config.dict())

        nxentry[map_config.sample.name] = NXsample(**map_config.sample.dict())

        nxentry.attrs['station'] = map_config.station
        
        nxentry.spec_scans = NXcollection()
        for scans in map_config.spec_scans:
            nxentry.spec_scans[scans.scanparsers[0].scan_name] = \
                NXfield(value=scans.scan_numbers,
                        dtype='int8',
                        attrs={'spec_file':str(scans.spec_file)})

        nxentry.data = NXdata()
        nxentry.data.attrs['axes'] = map_config.dims
        for i,dim in enumerate(map_config.independent_dimensions[::-1]):
            nxentry.data[dim.label] = NXfield(value=map_config.coords[dim.label],
                                              units=dim.units,
                                              attrs={'long_name': f'{dim.label} ({dim.units})', 
                                                     'data_type': dim.data_type,
                                                     'local_name': dim.name})
            nxentry.data.attrs[f'{dim.label}_indices'] = i

        signal = False
        auxilliary_signals = []
        for data in map_config.all_scalar_data:
            nxentry.data[data.label] = NXfield(value=np.empty(map_config.shape),
                                               units=data.units,
                                               attrs={'long_name': f'{data.label} ({data.units})',
                                                      'data_type': data.data_type,
                                                      'local_name': data.name})
            if not signal:
                signal = data.label
            else:
                auxilliary_signals.append(data.label)

        if signal:
            nxentry.data.attrs['signal'] = signal
            nxentry.data.attrs['auxilliary_signals'] = auxilliary_signals

        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(scan_number, scan_step_index, map_config)
                    for data in map_config.all_scalar_data:
                        nxentry.data[data.label][map_index] = data.get_value(scans, scan_number, scan_step_index)

        return(nxentry)

class IntegrationProcessor(Processor):
    '''Class representing a process that takes a map and integration
    configuration and returns a `nexusformat.nexus.NXprocess` containing a map of
    the integrated detector data requested.
    '''

    def process(self, data):
        '''Process the output of a `Reader` that contains a map and integration
        configuration and return a `nexusformat.nexus.NXprocess` containing a map
        of the integrated detector data requested

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key, and at least one item has
            the value `'IntegrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :return: integrated data and process metadata
        :rtype: nexusformat.nexus.NXprocess
        '''
        print(f'{self.__name__}: get MapConfig and IntegrationConfig from input, return an NXprocess')

        map_config, integration_config = self.get_configs(data)
        nxprocess = self.get_nxprocess(map_config, integration_config)

        return(nxprocess)

    def get_configs(self, data):
        '''Return valid instances of `MapConfig` and `IntegrationConfig` from the
        input supplied by `MultipleReader`.

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key, and at least one item has
            the value `'IntegrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises ValueError: if `data` cannot be parsed into map and integration configurations.
        :return: valid map and integration configuration objects.
        :rtype: tuple[MapConfig, IntegrationConfig]
        '''

        from models.map import MapConfig
        from models.integration import IntegrationConfig

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

    def get_nxprocess(self, map_config, integration_config):
        '''Use a `MapConfig` and `IntegrationConfig` to construct a
        `nexusformat.nexus.NXprocess`

        :param map_config: a valid map configuration
        :type map_config: MapConfig
        :param integration_config: a valid integration configuration
        :type integration_config" IntegrationConfig
        :return: the integrated detector data and metadata contained in a NeXus
            structure
        :rtype: nexusformat.nexus.NXprocess
        '''

        from nexusformat.nexus import (NXdata,
                                       NXdetector,
                                       NXfield,
                                       NXprocess)
        import numpy as np
        import pyFAI

        nxprocess = NXprocess(name=integration_config.title)

        nxprocess.map_config = json.dumps(map_config.dict())
        nxprocess.integration_config = json.dumps(integration_config.dict())

        nxprocess.program = 'pyFAI'
        nxprocess.version = pyFAI.version

        for k,v in integration_config.dict().items():
            if k == 'detectors': 
                continue
            nxprocess.attrs[k] = v

        for detector in integration_config.detectors:
            nxprocess[detector.prefix] = NXdetector()
            nxprocess[detector.prefix].local_name = detector.prefix
            nxprocess[detector.prefix].distance = detector.azimuthal_integrator.dist
            nxprocess[detector.prefix].distance.attrs['units'] = 'm'
            nxprocess[detector.prefix].calibration_wavelength = detector.azimuthal_integrator.wavelength
            nxprocess[detector.prefix].calibration_wavelength.attrs['units'] = 'm'
            nxprocess[detector.prefix].attrs['poni_file'] = str(detector.poni_file)
            nxprocess[detector.prefix].attrs['mask_file'] = str(detector.mask_file)
            nxprocess[detector.prefix].raw_data_files = np.full(map_config.shape, '', dtype='|S256')

        nxprocess.data = NXdata()

        nxprocess.data.attrs['axes'] = (*map_config.dims, *integration_config.integrated_data_dims)
        for i,dim in enumerate(map_config.independent_dimensions[::-1]):
            nxprocess.data[dim.label] = NXfield(value=map_config.coords[dim.label],
                                              units=dim.units,
                                              attrs={'long_name': f'{dim.label} ({dim.units})', 
                                                     'data_type': dim.data_type,
                                                     'local_name': dim.name})
            nxprocess.data.attrs[f'{dim.label}_indices'] = i

        for i,(coord_name,coord_values) in enumerate(integration_config.integrated_data_coordinates.items()):
            if coord_name == 'radial':
                type_ = pyFAI.units.RADIAL_UNITS
            elif coord_name == 'azimuthal':
                type_ = pyFAI.units.AZIMUTHAL_UNITS
            coord_units = pyFAI.units.to_unit(getattr(integration_config, f'{coord_name}_units'), type_=type_)
            nxprocess.data[coord_units.name] = coord_values
            nxprocess.data.attrs[f'{coord_units.name}_indices'] = i+len(map_config.coords)
            nxprocess.data[coord_units.name].units = coord_units.unit_symbol
            nxprocess.data[coord_units.name].attrs['long_name'] = coord_units.label

        nxprocess.data.attrs['signal'] = 'I'
        nxprocess.data.I = NXfield(value=np.empty((*tuple([len(coord_values) for coord_name,coord_values in map_config.coords.items()][::-1]), *integration_config.integrated_data_shape)),
                                   units='a.u',
                                   attrs={'long_name':'Intensity (a.u)'})

        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(scan_number, scan_step_index, map_config)
                    nxprocess.data.I[map_index] = integration_config.get_integrated_data(scans, scan_number, scan_step_index)
                    for detector in integration_config.detectors:
                        nxprocess[detector.prefix].raw_data_files[map_index] = scanparser.get_detector_data_file(detector.prefix, scan_step_index)

        return(nxprocess)

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

        print(f'{self.__name__}: get MCACeriaCalibrationConfig from input, return calibrated values')

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

        print(f'{self.__name__}: get MapConfig and MCACeriaCalibrationConfig from input, return map of calibrated MCA data')

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

        from models.map import MapConfig

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

        print(f'{self.__name__}: get StrainAnalysisConfig from input, return map of strains')

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
