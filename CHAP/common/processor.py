#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Processors used in multiple experiment-specific
             workflows.
"""

# system modules
from json import dumps
from time import time

# local modules
from CHAP import Processor


class AsyncProcessor(Processor):
    """A Processor to process multiple sets of input data via asyncio
    module

    :ivar mgr: The `Processor` used to process every set of input data
    :type mgr: Processor
    """
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr

    def process(self, data):
        """Asynchronously process the input documents with the
        `self.mgr` `Processor`.

        :param data: input data documents to process
        :type docs: iterable
        """

        import asyncio

        async def task(mgr, doc):
            """Process given data using provided `Processor`

            :param mgr: the object that will process given data
            :type mgr: Processor
            :param doc: the data to process
            :type doc: object
            :return: processed data
            :rtype: object
            """
            return mgr.process(doc)

        async def execute_tasks(mgr, docs):
            """Process given set of documents using provided task
            manager

            :param mgr: the object that will process all documents
            :type mgr: Processor
            :param docs: the set of data documents to process
            :type doc: iterable
            """
            coroutines = [task(mgr, d) for d in docs]
            await asyncio.gather(*coroutines)

        asyncio.run(execute_tasks(self.mgr, data))


class IntegrationProcessor(Processor):
    """A processor for integrating 2D data with pyFAI"""

    def process(self, data):
        """Integrate the input data with the integration method and
        keyword arguments supplied and return the results.

        :param data: input data, including raw data, integration
            method, and keyword args for the integration method.
        :type data: tuple[typing.Union[numpy.ndarray,
                          list[numpy.ndarray]], callable, dict]
        :param integration_method: the method of a
            `pyFAI.azimuthalIntegrator.AzimuthalIntegrator` or
            `pyFAI.multi_geometry.MultiGeometry` that returns the
            desired integration results.
        :return: integrated raw data
        :rtype: pyFAI.containers.IntegrateResult
        """
        detector_data, integration_method, integration_kwargs = data

        return integration_method(detector_data, **integration_kwargs)


class IntegrateMapProcessor(Processor):
    """Class representing a process that takes a map and integration
    configuration and returns a `nexusformat.nexus.NXprocess`
    containing a map of the integrated detector data requested.
    """

    def process(self, data):
        """Process the output of a `Reader` that contains a map and
        integration configuration and return a
        `nexusformat.nexus.NXprocess` containing a map of the
        integrated detector data requested

        :param data: Result of `Reader.read` where at least one item
            has the value `'MapConfig'` for the `'schema'` key, and at
            least one item has the value `'IntegrationConfig'` for the
            `'schema'` key.
        :type data: list[dict[str,object]]
        :return: integrated data and process metadata
        :rtype: nexusformat.nexus.NXprocess
        """

        map_config, integration_config = self.get_configs(data)
        nxprocess = self.get_nxprocess(map_config, integration_config)

        return nxprocess

    def get_configs(self, data):
        """Return valid instances of `MapConfig` and
        `IntegrationConfig` from the input supplied by
        `MultipleReader`.

        :param data: Result of `Reader.read` where at least one item
            has the value `'MapConfig'` for the `'schema'` key, and at
            least one item has the value `'IntegrationConfig'` for the
            `'schema'` key.
        :type data: list[dict[str,object]]
        :raises ValueError: if `data` cannot be parsed into map and
            integration configurations.
        :return: valid map and integration configuration objects.
        :rtype: tuple[MapConfig, IntegrationConfig]
        """

        self.logger.debug('Getting configuration objects')
        t0 = time()

        from CHAP.common.models.map import MapConfig
        from CHAP.common.models.integration import IntegrationConfig

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
            raise ValueError('No map configuration found')
        if not integration_config:
            raise ValueError('No integration configuration found')

        map_config = MapConfig(**map_config)
        integration_config = IntegrationConfig(**integration_config)

        self.logger.debug(
            f'Got configuration objects in {time()-t0:.3f} seconds')

        return map_config, integration_config

    def get_nxprocess(self, map_config, integration_config):
        """Use a `MapConfig` and `IntegrationConfig` to construct a
        `nexusformat.nexus.NXprocess`

        :param map_config: a valid map configuration
        :type map_config: MapConfig
        :param integration_config: a valid integration configuration
        :type integration_config: IntegrationConfig
        :return: the integrated detector data and metadata contained
            in a NeXus structure
        :rtype: nexusformat.nexus.NXprocess
        """

        self.logger.debug('Constructing NXprocess')
        t0 = time()

        from nexusformat.nexus import (NXdata,
                                       NXdetector,
                                       NXfield,
                                       NXprocess)
        import numpy as np
        import pyFAI

        nxprocess = NXprocess(name=integration_config.title)

        nxprocess.map_config = dumps(map_config.dict())
        nxprocess.integration_config = dumps(integration_config.dict())

        nxprocess.program = 'pyFAI'
        nxprocess.version = pyFAI.version

        for k, v in integration_config.dict().items():
            if k == 'detectors':
                continue
            nxprocess.attrs[k] = v

        for detector in integration_config.detectors:
            nxprocess[detector.prefix] = NXdetector()
            nxdetector = nxprocess[detector.prefix]
            nxdetector.local_name = detector.prefix
            nxdetector.distance = detector.azimuthal_integrator.dist
            nxdetector.distance.attrs['units'] = 'm'
            nxdetector.calibration_wavelength = \
                detector.azimuthal_integrator.wavelength
            nxdetector.calibration_wavelength.attrs['units'] = 'm'
            nxdetector.attrs['poni_file'] = str(detector.poni_file)
            nxdetector.attrs['mask_file'] = str(detector.mask_file)
            nxdetector.raw_data_files = np.full(map_config.shape,
                                                '', dtype='|S256')

        nxprocess.data = NXdata()

        nxprocess.data.attrs['axes'] = (
            *map_config.dims,
            *integration_config.integrated_data_dims
        )
        for i, dim in enumerate(map_config.independent_dimensions[::-1]):
            nxprocess.data[dim.label] = NXfield(
                value=map_config.coords[dim.label],
                units=dim.units,
                attrs={'long_name': f'{dim.label} ({dim.units})',
                       'data_type': dim.data_type,
                       'local_name': dim.name})
            nxprocess.data.attrs[f'{dim.label}_indices'] = i

        for i, (coord_name, coord_values) in enumerate(
                integration_config.integrated_data_coordinates.items()):
            if coord_name == 'radial':
                type_ = pyFAI.units.RADIAL_UNITS
            elif coord_name == 'azimuthal':
                type_ = pyFAI.units.AZIMUTHAL_UNITS
            coord_units = pyFAI.units.to_unit(
                getattr(integration_config, f'{coord_name}_units'),
                type_=type_)
            nxprocess.data[coord_units.name] = coord_values
            nxprocess.data.attrs[f'{coord_units.name}_indices'] = i + len(
                map_config.coords)
            nxprocess.data[coord_units.name].units = coord_units.unit_symbol
            nxprocess.data[coord_units.name].attrs['long_name'] = \
                coord_units.label

        nxprocess.data.attrs['signal'] = 'I'
        nxprocess.data.I = NXfield(
            value=np.empty(
                (*tuple(
                    [len(coord_values) for coord_name, coord_values
                     in map_config.coords.items()][::-1]),
                 *integration_config.integrated_data_shape)),
            units='a.u',
            attrs={'long_name':'Intensity (a.u)'})

        integrator = integration_config.get_multi_geometry_integrator()
        if integration_config.integration_type == 'azimuthal':
            integration_method = integrator.integrate1d
            integration_kwargs = {
                'lst_mask': [detector.mask_array
                             for detector
                             in integration_config.detectors],
                'npt': integration_config.radial_npt
            }
        elif integration_config.integration_type == 'cake':
            integration_method = integrator.integrate2d
            integration_kwargs = {
                'lst_mask': [detector.mask_array
                             for detector
                             in integration_config.detectors],
                'npt_rad': integration_config.radial_npt,
                'npt_azim': integration_config.azimuthal_npt,
                'method': 'bbox'
            }

        integration_processor = IntegrationProcessor()
        integration_processor.logger.setLevel(self.logger.getEffectiveLevel())
        for handler in self.logger.handlers:
            integration_processor.logger.addHandler(handler)
        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(
                        scan_number,
                        scan_step_index,
                        map_config)
                    detector_data = scans.get_detector_data(
                        integration_config.detectors,
                        scan_number,
                        scan_step_index)
                    result = integration_processor.process(
                        (detector_data,
                         integration_method, integration_kwargs))
                    nxprocess.data.I[map_index] = result.intensity

                    for detector in integration_config.detectors:
                        nxprocess[detector.prefix].raw_data_files[map_index] =\
                            scanparser.get_detector_data_file(
                                detector.prefix, scan_step_index)

        self.logger.debug(f'Constructed NXprocess in {time()-t0:.3f} seconds')

        return nxprocess


class MapProcessor(Processor):
    """A Processor to take a map configuration and return a
    `nexusformat.nexus.NXentry` representing that map's metadata and
    any scalar-valued raw data requseted by the supplied map
    configuration.
    """

    def process(self, data):
        """Process the output of a `Reader` that contains a map
        configuration and return a `nexusformat.nexus.NXentry`
        representing the map.

        :param data: Result of `Reader.read` where at least one item
            has the value `'MapConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :return: Map data & metadata
        :rtype: nexusformat.nexus.NXentry
        """

        map_config = self.get_map_config(data)
        nxentry = self.__class__.get_nxentry(map_config)

        return nxentry

    def get_map_config(self, data):
        """Get an instance of `MapConfig` from a returned value of
        `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'MapConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid `MapConfig` cannot be
            constructed from `data`.
        :return: a valid instance of `MapConfig` with field values
            taken from `data`.
        :rtype: MapConfig
        """

        from CHAP.common.models.map import MapConfig

        map_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'MapConfig':
                        map_config = item.get('data')
                        break

        if not map_config:
            raise ValueError('No map configuration found')

        return MapConfig(**map_config)

    @staticmethod
    def get_nxentry(map_config):
        """Use a `MapConfig` to construct a
        `nexusformat.nexus.NXentry`

        :param map_config: a valid map configuration
        :type map_config: MapConfig
        :return: the map's data and metadata contained in a NeXus
            structure
        :rtype: nexusformat.nexus.NXentry
        """

        from nexusformat.nexus import (NXcollection,
                                       NXdata,
                                       NXentry,
                                       NXfield,
                                       NXsample)
        import numpy as np

        nxentry = NXentry(name=map_config.title)

        nxentry.map_config = dumps(map_config.dict())

        nxentry[map_config.sample.name] = NXsample(**map_config.sample.dict())

        nxentry.attrs['station'] = map_config.station

        nxentry.spec_scans = NXcollection()
        for scans in map_config.spec_scans:
            nxentry.spec_scans[scans.scanparsers[0].scan_name] = \
                NXfield(value=scans.scan_numbers,
                        dtype='int8',
                        attrs={'spec_file': str(scans.spec_file)})

        nxentry.data = NXdata()
        nxentry.data.attrs['axes'] = map_config.dims
        for i, dim in enumerate(map_config.independent_dimensions[::-1]):
            nxentry.data[dim.label] = NXfield(
                value=map_config.coords[dim.label],
                units=dim.units,
                attrs={'long_name': f'{dim.label} ({dim.units})',
                       'data_type': dim.data_type,
                       'local_name': dim.name})
            nxentry.data.attrs[f'{dim.label}_indices'] = i

        signal = False
        auxilliary_signals = []
        for data in map_config.all_scalar_data:
            nxentry.data[data.label] = NXfield(
                value=np.empty(map_config.shape),
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
                    map_index = scans.get_index(
                        scan_number,
                        scan_step_index,
                        map_config)
                    for data in map_config.all_scalar_data:
                        nxentry.data[data.label][map_index] = data.get_value(
                            scans,
                            scan_number,
                            scan_step_index)

        return nxentry


class NexusToNumpyProcessor(Processor):
    """A Processor to convert the default plottable data in an
    `NXobject` into an `numpy.ndarray`.
    """

    def process(self, data):
        """Return the default plottable data signal in `data` as an
        `numpy.ndarray`.

        :param data: input NeXus structure
        :type data: nexusformat.nexus.tree.NXobject
        :raises ValueError: if `data` has no default plottable data
            signal
        :return: default plottable data signal in `data`
        :rtype: numpy.ndarray
        """

        from nexusformat.nexus import NXdata

        data = self.unwrap_pipelinedata(data)

        if isinstance(data, NXdata):
            default_data = data
        else:
            default_data = data.plottable_data
            if default_data is None:
                default_data_path = data.attrs.get('default')
                default_data = data.get(default_data_path)
            if default_data is None:
                raise ValueError(
                    f'The structure of {data} contains no default data')

        default_signal = default_data.attrs.get('signal')
        if default_signal is None:
            raise ValueError(f'The signal of {default_data} is unknown')
        default_signal = default_signal.nxdata

        np_data = default_data[default_signal].nxdata

        return np_data


class NexusToXarrayProcessor(Processor):
    """A Processor to convert the default plottable data in an
    `NXobject` into an `xarray.DataArray`.
    """

    def process(self, data):
        """Return the default plottable data signal in `data` as an
        `xarray.DataArray`.

        :param data: input NeXus structure
        :type data: nexusformat.nexus.tree.NXobject
        :raises ValueError: if metadata for `xarray` is absent from
            `data`
        :return: default plottable data signal in `data`
        :rtype: xarray.DataArray
        """

        from nexusformat.nexus import NXdata
        from xarray import DataArray

        data = self.unwrap_pipelinedata(data)

        if isinstance(data, NXdata):
            default_data = data
        else:
            default_data = data.plottable_data
            if default_data is None:
                default_data_path = data.attrs.get('default')
                default_data = data.get(default_data_path)
            if default_data is None:
                raise ValueError(
                    f'The structure of {data} contains no default data')

        default_signal = default_data.attrs.get('signal')
        if default_signal is None:
            raise ValueError(f'The signal of {default_data} is unknown')
        default_signal = default_signal.nxdata

        signal_data = default_data[default_signal].nxdata

        axes = default_data.attrs['axes']
        if isinstance(axes, str):
            axes = [axes]
        coords = {}
        for axis_name in axes:
            axis = default_data[axis_name]
            coords[axis_name] = (axis_name,
                                 axis.nxdata,
                                 axis.attrs)

        dims = tuple(axes)
        name = default_signal
        attrs = default_data[default_signal].attrs

        return DataArray(data=signal_data,
                         coords=coords,
                         dims=dims,
                         name=name,
                         attrs=attrs)


class PrintProcessor(Processor):
    """A Processor to simply print the input data to stdout and return
    the original input data, unchanged in any way.
    """

    def process(self, data):
        """Print and return the input data.

        :param data: Input data
        :type data: object
        :return: `data`
        :rtype: object
        """

        print(f'{self.__name__} data :')

        if callable(getattr(data, '_str_tree', None)):
            # If data is likely an NXobject, print its tree
            # representation (since NXobjects' str representations are
            # just their nxname)
            print(data._str_tree(attrs=True, recursive=True))
        else:
            print(str(data))

        return data


class StrainAnalysisProcessor(Processor):
    """A Processor to compute a map of sample strains by fitting bragg
    peaks in 1D detector data and analyzing the difference between
    measured peak locations and expected peak locations for the sample
    measured.
    """

    def process(self, data):
        """Process the input map detector data & configuration for the
        strain analysis procedure, and return a map of sample strains.

        :param data: results of `MutlipleReader.read` containing input
            map detector data and strain analysis configuration
        :type data: dict[list[str,object]]
        :return: map of sample strains
        :rtype: xarray.Dataset
        """

        strain_analysis_config = self.get_config(data)

        return data

    def get_config(self, data):
        """Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'StrainAnalysisConfig'` for the `'schema'`
            key.
        :type data: list[dict[str,object]]
        :raises Exception: If valid config objects cannot be
            constructed from `data`.
        :return: valid instances of the configuration objects with
            field values taken from `data`.
        :rtype: StrainAnalysisConfig
        """

        strain_analysis_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'StrainAnalysisConfig':
                        strain_analysis_config = item.get('data')

        if not strain_analysis_config:
            raise ValueError(
                'No strain analysis configuration found in input data')

        return strain_analysis_config


class XarrayToNexusProcessor(Processor):
    """A Processor to convert the data in an `xarray` structure to an
    `nexusformat.nexus.NXdata`.
    """

    def process(self, data):
        """Return `data` represented as an `nexusformat.nexus.NXdata`.

        :param data: The input `xarray` structure
        :type data: typing.Union[xarray.DataArray, xarray.Dataset]
        :return: The data and metadata in `data`
        :rtype: nexusformat.nexus.NXdata
        """

        from nexusformat.nexus import NXdata, NXfield

        data = self.unwrap_pipelinedata(data)

        signal = NXfield(value=data.data, name=data.name, attrs=data.attrs)

        axes = []
        for name, coord in data.coords.items():
            axes.append(
                NXfield(value=coord.data, name=name, attrs=coord.attrs))
        axes = tuple(axes)

        return NXdata(signal=signal, axes=axes)


class XarrayToNumpyProcessor(Processor):
    """A Processor to convert the data in an `xarray.DataArray`
    structure to an `numpy.ndarray`.
    """

    def process(self, data):
        """Return just the signal values contained in `data`.

        :param data: The input `xarray.DataArray`
        :type data: xarray.DataArray
        :return: The data in `data`
        :rtype: numpy.ndarray
        """

        return self.unwrap_pipelinedata(data).data


if __name__ == '__main__':
    from CHAP.processor import main
    main()
