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
import logging
import sys
from time import time

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
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def process(self, data):
        """
        process data API
        """

        t0 = time()
        self.logger.info(f'Executing "process" with type(data)={type(data)}')

        data = self._process(data)

        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')

        return(data)

    def _process(self, data):
        # If needed, extract data from a returned value of Reader.read
        if isinstance(data, list):
            if all([isinstance(d,dict) for d in data]):
                data = data[0]['data']
        # process operation is a simple print function
        data += "process part\n"
        # and we return data back to pipeline
        return data


class TFaaSImageProcessor(Processor):
    '''
    A Processor to get predictions from TFaaS inference server.
    '''
    def process(self, data, url, model, verbose=False):
        """
        process data API
        """

        t0 = time()
        self.logger.info(f'Executing "process" with url {url} model {model}')

        data = self._process(data, url, model, verbose)

        self.logger.info(f'Finished "process" in {time()-t0:.3f} seconds\n')

        return(data)

    def _process(self, data, url, model, verbose):
        '''Print and return the input data.

        :param data: Input image data, either file name or actual image data
        :type data: object
        :return: `data`
        :rtype: object
        '''
        from MLaaS.tfaas_client import predictImage
        from pathlib import Path
        self.logger.info(f"input data {type(data)}")
        if isinstance(data, str) and Path(data).is_file():
            imgFile = data
            data = predictImage(url, imgFile, model, verbose)
        else:
            rdict = data[0]
            import requests
            img = rdict['data']
            session = requests.Session()
            rurl = url + '/predict/image'
            payload = dict(model=model)
            files = dict(image=img)
            self.logger.info(f"HTTP request {rurl} with image file and {payload} payload")
            req = session.post(rurl, files=files, data=payload )
            data = req.content
            data = data.decode("utf-8").replace('\n', '')
            self.logger.info(f"HTTP response {data}")

        return(data)

class URLResponseProcessor(Processor):
    def _process(self, data):
        '''Take data returned from URLReader.read and return a decoded version of
        the content.

        :param data: input data (output of URLReader.read)
        :type data: list[dict]
        :return: decoded data contents
        :rtype: object
        '''

        data = data[0]

        content = data['data']
        encoding = data['encoding']

        self.logger.debug(f'Decoding content of type {type(content)} with {encoding}')

        try:
            content = content.decode(encoding)
        except:
            self.logger.warning(f'Failed to decode content of type {type(content)} with {encoding}')

        return(content)

class PrintProcessor(Processor):
    '''A Processor to simply print the input data to stdout and return the
    original input data, unchanged in any way.
    '''

    def _process(self, data):
        '''Print and return the input data.

        :param data: Input data
        :type data: object
        :return: `data`
        :rtype: object
        '''

        print(f'{self.__name__} data :')

        if callable(getattr(data, '_str_tree', None)):
            # If data is likely an NXobject, print its tree representation
            # (since NXobjects' str representations are just their nxname -- not
            # very helpful).
            print(data._str_tree(attrs=True, recursive=True))
        else:
            print(str(data))

        return(data)

class NexusToNumpyProcessor(Processor):
    '''A class to convert the default plottable data in an `NXobject` into an
    `numpy.ndarray`.
    '''

    def _process(self, data):
        '''Return the default plottable data signal in `data` as an
        `numpy.ndarray`.

        :param data: input NeXus structure
        :type data: nexusformat.nexus.tree.NXobject
        :raises ValueError: if `data` has no default plottable data signal
        :return: default plottable data signal in `data`
        :rtype: numpy.ndarray
        '''

        default_data = data.plottable_data

        if default_data is None:
            default_data_path = data.attrs['default']
            default_data = data.get(default_data_path)
        if default_data is None:
            raise(ValueError(f'The structure of {data} contains no default data'))

        default_signal = default_data.attrs.get('signal')
        if default_signal is None:
            raise(ValueError(f'The signal of {default_data} is unknown'))
        default_signal = default_signal.nxdata

        np_data = default_data[default_signal].nxdata

        return(np_data)

class NexusToXarrayProcessor(Processor):
    '''A class to convert the default plottable data in an `NXobject` into an
    `xarray.DataArray`.'''

    def _process(self, data):
        '''Return the default plottable data signal in `data` as an
        `xarray.DataArray`.

        :param data: input NeXus structure
        :type data: nexusformat.nexus.tree.NXobject
        :raises ValueError: if metadata for `xarray` is absen from `data`
        :return: default plottable data signal in `data`
        :rtype: xarray.DataArray
        '''

        from xarray import DataArray

        default_data = data.plottable_data

        if default_data is None:
            default_data_path = data.attrs['default']
            default_data = data.get(default_data_path)
        if default_data is None:
            raise(ValueError(f'The structure of {data} contains no default data'))

        default_signal = default_data.attrs.get('signal')
        if default_signal is None:
            raise(ValueError(f'The signal of {default_data} is unknown'))
        default_signal = default_signal.nxdata

        signal_data = default_data[default_signal].nxdata

        axes = default_data.attrs['axes']
        coords = {}
        for axis_name in axes:
            axis = default_data[axis_name]
            coords[axis_name] = (axis_name,
                                 axis.nxdata,
                                 axis.attrs)

        dims = tuple(axes)

        name = default_signal

        attrs = default_data[default_signal].attrs

        return(DataArray(data=signal_data,
                         coords=coords,
                         dims=dims,
                         name=name,
                         attrs=attrs))

class XarrayToNexusProcessor(Processor):
    '''A class to convert the data in an `xarray` structure to an
    `nexusformat.nexus.NXdata`.
    '''

    def _process(self, data):
        '''Return `data` represented as an `nexusformat.nexus.NXdata`.

        :param data: The input `xarray` structure
        :type data: typing.Union[xarray.DataArray, xarray.Dataset]
        :return: The data and metadata in `data`
        :rtype: nexusformat.nexus.NXdata
        '''

        from nexusformat.nexus import NXdata, NXfield

        signal = NXfield(value=data.data, name=data.name, attrs=data.attrs)

        axes = []
        for name, coord in data.coords.items():
            axes.append(NXfield(value=coord.data, name=name, attrs=coord.attrs))
        axes = tuple(axes)

        return(NXdata(signal=signal, axes=axes))

class XarrayToNumpyProcessor(Processor):
    '''A class to convert the data in an `xarray.DataArray` structure to an
    `numpy.ndarray`.
    '''

    def _process(self, data):
        '''Return just the signal values contained in `data`.

        :param data: The input `xarray.DataArray`
        :type data: xarray.DataArray
        :return: The data in `data`
        :rtype: numpy.ndarray
        '''

        return(data.data)

class MapProcessor(Processor):
    '''Class representing a process that takes a map configuration and returns a
    `nexusformat.nexus.NXentry` representing that map's metadata and any
    scalar-valued raw data requseted by the supplied map configuration.
    '''

    def _process(self, data):
        '''Process the output of a `Reader` that contains a map configuration and
        return a `nexusformat.nexus.NXentry` representing the map.

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :return: Map data & metadata (SPEC only, no detector)
        :rtype: nexusformat.nexus.NXentry
        '''

        map_config = self.get_map_config(data)
        nxentry = self.__class__.get_nxentry(map_config)

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

        from CHAP.models.map import MapConfig

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
        
    @staticmethod
    def get_nxentry(map_config):
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
    '''Class for integrating 2D detector data
    '''

    def _process(self, data):
        '''Integrate the input data with the integration method and keyword
        arguments supplied and return the results.

        :param data: input data, including raw data, integration method, and
            keyword args for the integration method.
        :type data: tuple[typing.Union[numpy.ndarray, list[numpy.ndarray]],
                          callable,
                          dict]
        :param integration_method: the method of a
            `pyFAI.azimuthalIntegrator.AzimuthalIntegrator` or
            `pyFAI.multi_geometry.MultiGeometry` that returns the desired
            integration results.
        :return: integrated raw data
        :rtype: pyFAI.containers.IntegrateResult
        '''

        detector_data, integration_method, integration_kwargs = data

        return(integration_method(detector_data, **integration_kwargs))

class IntegrateMapProcessor(Processor):
    '''Class representing a process that takes a map and integration
    configuration and returns a `nexusformat.nexus.NXprocess` containing a map of
    the integrated detector data requested.
    '''

    def _process(self, data):
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

        self.logger.debug('Getting configuration objects')
        t0 = time()

        from CHAP.models.map import MapConfig
        from CHAP.models.integration import IntegrationConfig

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

        map_config = MapConfig(**map_config)
        integration_config = IntegrationConfig(**integration_config)

        self.logger.debug(f'Got configuration objects in {time()-t0:.3f} seconds')

        return(map_config, integration_config)

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

        self.logger.debug('Constructing NXprocess')
        t0 = time()

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

        integrator = integration_config.get_multi_geometry_integrator()
        if integration_config.integration_type == 'azimuthal':
            integration_method = integrator.integrate1d
            integration_kwargs = {
                'lst_mask': [detector.mask_array for detector in integration_config.detectors],
                'npt': integration_config.radial_npt
            }
        elif integration_config.integration_type == 'cake':
            integration_method = integrator.integrate2d
            integration_kwargs = {
                'lst_mask': [detector.mask_array for detector in integration_config.detectors],
                'npt_rad': integration_config.radial_npt,
                'npt_azim': integration_config.azimuthal_npt,
                'method': 'bbox'
            }

        integration_processor = IntegrationProcessor()
        integration_processor.logger.setLevel(self.logger.getEffectiveLevel())
        integration_processor.logger.addHandler(self.logger.handlers[0])
        lst_args = []
        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(scan_number, scan_step_index, map_config)
                    detector_data = scans.get_detector_data(integration_config.detectors, scan_number, scan_step_index)
                    result = integration_processor.process((detector_data, integration_method, integration_kwargs))
                    nxprocess.data.I[map_index] = result.intensity
                    for detector in integration_config.detectors:
                        nxprocess[detector.prefix].raw_data_files[map_index] = scanparser.get_detector_data_file(detector.prefix, scan_step_index)

        self.logger.debug(f'Constructed NXprocess in {time()-t0:.3f} seconds')

        return(nxprocess)

class MCACeriaCalibrationProcessor(Processor):
    '''Class representing the procedure to use a CeO2 scan to obtain tuned values
    for the bragg diffraction angle and linear correction parameters for MCA
    channel energies for an EDD experimental setup.
    '''

    def _process(self, data):
        '''Return tuned values for 2&theta and linear correction parameters for
        the MCA channel energies.

        :param data: input configuration for the raw data & tuning procedure
        :type data: list[dict[str,object]]
        :return: original configuration dictionary with tuned values added
        :rtype: dict[str,float]
        '''

        calibration_config = self.get_config(data)

        tth, slope, intercept = self.calibrate(calibration_config)

        calibration_config.tth_calibrated = tth
        calibration_config.slope_calibrated = slope
        calibration_config.intercept_calibrated = intercept

        return(calibration_config.dict())

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

        from CHAP.models.edd import MCACeriaCalibrationConfig

        calibration_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'MCACeriaCalibrationConfig':
                        calibration_config = item.get('data')
                        break

        if not calibration_config:
            raise(ValueError('No MCA ceria calibration configuration found in input data'))

        return(MCACeriaCalibrationConfig(**calibration_config))

    def calibrate(self, calibration_config):
        '''Iteratively calibrate 2&theta by fitting selected peaks of an MCA
        spectrum until the computed strain is sufficiently small. Use the fitted
        peak locations to determine linear correction parameters for the MCA's
        channel energies.

        :param calibration_config: object configuring the CeO2 calibration procedure
        :type calibration_config: MCACeriaCalibrationConfig
        :return: calibrated values of 2&theta and linear correction parameters
            for MCA channel energies : tth, slope, intercept
        :rtype: float, float, float
        '''

        from msnctools.fit import Fit, FitMultipeak
        import numpy as np
        from scipy.constants import physical_constants

        hc = physical_constants['Planck constant in eV/Hz'][0] * \
             physical_constants['speed of light in vacuum'][0] * \
             1e7 # We'll work in keV and A, not eV and m.

        # Collect raw MCA data of interest
        mca_data = calibration_config.mca_data()
        mca_bin_energies = np.arange(0, calibration_config.num_bins) * \
                           (calibration_config.max_energy_kev / calibration_config.num_bins)

        # Mask out the corrected MCA data for fitting
        mca_mask = calibration_config.mca_mask()
        fit_mca_energies = mca_bin_energies[mca_mask]
        fit_mca_intensities = mca_data[mca_mask]

        # Correct raw MCA data for variable flux at different energies
        flux_correct = calibration_config.flux_correction_interpolation_function()
        mca_intensity_weights = flux_correct(fit_mca_energies)
        fit_mca_intensities = fit_mca_intensities / mca_intensity_weights

        # Get the HKLs and lattice spacings that will be used for fitting
        tth = calibration_config.tth_initial_guess
        fit_hkls, fit_ds = calibration_config.fit_ds()
        c_1 = fit_hkls[:,0]**2 + fit_hkls[:,1]**2 + fit_hkls[:,2]**2

        for iter_i in range(calibration_config.max_iter):

            ### Perform the uniform fit first ###

            # Get expected peak energy locations for this iteration's starting
            # value of tth
            fit_lambda = 2.0 * fit_ds * np.sin(0.5*np.radians(tth))
            fit_E0 = hc / fit_lambda

            # Run the uniform fit
            best_fit, residual, best_values, best_errors, redchi, success = \
                FitMultipeak.fit_multipeak(fit_mca_intensities,
                                           fit_E0,
                                           x=fit_mca_energies,
                                           fit_type='uniform')

            # Extract values of interest from the best values for the uniform fit
            # parameters
            uniform_fit_centers = [best_values[f'peak{i+1}_center'] for i in range(len(calibration_config.fit_hkls))]
            # uniform_a = best_values['scale_factor']
            # uniform_strain = np.log(uniform_a / calibration_config.lattice_parameter_angstrom)
            # uniform_tth = tth * (1.0 + uniform_strain)
            # uniform_rel_rms_error = np.linalg.norm(residual) / np.linalg.norm(fit_mca_intensities)

            ### Next, perform the unconstrained fit ###

            # Use the peak locations found in the uniform fit as the initial
            # guesses for peak locations in the unconstrained fit
            best_fit, residual, best_values, best_errors, redchi, success = \
                FitMultipeak.fit_multipeak(fit_mca_intensities,
                                           uniform_fit_centers,
                                           x=fit_mca_energies,
                                           fit_type='unconstrained')

            # Extract values of interest from the best values for the
            # unconstrained fit parameters
            unconstrained_fit_centers = np.array([best_values[f'peak{i+1}_center'] for i in range(len(calibration_config.fit_hkls))])
            unconstrained_a = 0.5 * hc * np.sqrt(c_1) / (unconstrained_fit_centers * abs(np.sin(0.5*np.radians(tth))))
            unconstrained_strains = np.log(unconstrained_a / calibration_config.lattice_parameter_angstrom)
            unconstrained_strain = np.mean(unconstrained_strains)
            unconstrained_tth = tth * (1.0 + unconstrained_strain)
            # unconstrained_rel_rms_error = np.linalg.norm(residual) / np.linalg.norm(fit_mca_intensities)


            # Update tth for the next iteration of tuning
            prev_tth = tth
            tth = unconstrained_tth

            # Stop tuning tth at this iteration if differences are small enough
            if abs(tth - prev_tth) < calibration_config.tune_tth_tol:
                break

        # Fit line to expected / computed peak locations from the last
        # unconstrained fit.
        fit = Fit.fit_data(fit_E0,'linear', x=unconstrained_fit_centers, nan_policy='omit')
        slope = fit.best_values['slope']
        intercept = fit.best_values['intercept']

        return(float(tth), float(slope), float(intercept))

class MCADataProcessor(Processor):
    '''Class representing a process to return data from a MCA, restuctured to
    incorporate the shape & metadata associated with a map configuration to
    which the MCA data belongs, and linearly transformed according to the
    results of a ceria calibration.
    '''

    def _process(self, data):
        '''Process configurations for a map and MCA detector(s), and return the
        raw MCA data collected over the map.

        :param data: input map configuration and results of ceria calibration
        :type data: list[dict[str,object]]
        :return: calibrated and flux-corrected MCA data
        :rtype: nexusformat.nexus.NXentry
        '''

        map_config, calibration_config = self.get_configs(data)
        nxroot = self.get_nxroot(map_config, calibration_config)

        return(nxroot)

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

        from CHAP.models.map import MapConfig
        from CHAP.models.edd import MCACeriaCalibrationConfig

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

        return(MapConfig(**map_config), MCACeriaCalibrationConfig(**calibration_config))

    def get_nxroot(self, map_config, calibration_config):
        '''Get a map of the MCA data collected by the scans in `map_config`. The
        MCA data will be calibrated and flux-corrected according to the
        parameters included in `calibration_config`. The data will be returned
        along with relevant metadata in the form of a NeXus structure.

        :param map_config: the map configuration
        :type map_config: MapConfig
        :param calibration_config: the calibration configuration
        :type calibration_config: MCACeriaCalibrationConfig
        :return: a map of the calibrated and flux-corrected MCA data
        :rtype: nexusformat.nexus.NXroot
        '''

        from nexusformat.nexus import (NXdata,
                                       NXdetector,
                                       NXentry,
                                       NXinstrument,
                                       NXroot)
        import numpy as np

        nxroot = NXroot()

        nxroot[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxentry = nxroot[map_config.title]

        nxentry.instrument = NXinstrument()
        nxentry.instrument.detector = NXdetector()
        nxentry.instrument.detector.calibration_configuration = json.dumps(calibration_config.dict())

        nxentry.instrument.detector.data = NXdata()
        nxdata = nxentry.instrument.detector.data
        nxdata.raw = np.empty((*map_config.shape, calibration_config.num_bins))
        nxdata.raw.attrs['units'] = 'counts'
        nxdata.channel_energy = calibration_config.slope_calibrated * \
                                np.arange(0, calibration_config.num_bins) * \
                                (calibration_config.max_energy_kev / calibration_config.num_bins) + \
                                calibration_config.intercept_calibrated
        nxdata.channel_energy.attrs['units'] = 'keV'

        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(scan_number, scan_step_index, map_config)
                    nxdata.raw[map_index] = scanparser.get_detector_data(calibration_config.detector_name, scan_step_index)

        nxentry.data.makelink(nxdata.raw, name=calibration_config.detector_name)
        nxentry.data.makelink(nxdata.channel_energy, name=f'{calibration_config.detector_name}_channel_energy')
        if isinstance(nxentry.data.attrs['axes'], str):
            nxentry.data.attrs['axes'] = [nxentry.data.attrs['axes'], f'{calibration_config.detector_name}_channel_energy']
        else:
            nxentry.data.attrs['axes'] += [f'{calibration_config.detector_name}_channel_energy']
        nxentry.data.attrs['signal'] = calibration_config.detector_name

        return(nxroot)

class StrainAnalysisProcessor(Processor):
    '''Class representing a process to compute a map of sample strains by fitting
    bragg peaks in 1D detector data and analyzing the difference between measured
    peak locations and expected peak locations for the sample measured.
    '''

    def _process(self, data):
        '''Process the input map detector data & configuration for the strain
        analysis procedure, and return a map of sample strains.

        :param data: results of `MutlipleReader.read` containing input map
            detector data and strain analysis configuration
        :type data: dict[list[str,object]]
        :return: map of sample strains
        :rtype: xarray.Dataset
        '''

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
        self.parser.add_argument('--log-level', choices=logging._nameToLevel.keys(),
            dest='log_level', default='INFO', help='logging level')

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
    processor.logger.setLevel(getattr(logging, opts.log_level))
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter('{name:20}: {message}', style='{'))
    processor.logger.addHandler(log_handler)
    data = processor.process(opts.data)

    print(f"Processor {processor} operates on data {data}")

if __name__ == '__main__':
    main()
