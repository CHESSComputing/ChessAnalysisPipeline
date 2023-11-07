#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Processors used in multiple experiment-specific
             workflows.
"""

# Third party modules
import numpy as np

# Local modules
from CHAP import Processor


class AnimationProcessor(Processor):
    """A Processor to show and return an animation.
    """
    def process(
            self, data, num_frames, axis=0, interval=1000, blit=True,
            repeat=True, repeat_delay=1000):
        """Show and return an animation of image slices from a dataset
        contained in `data`.

        :param data: Input data.
        :type data: CHAP.pipeline.PipelineData
        :param num_frames: Number of frames for the animation.
        :type num_frames: int
        :param axis: Axis direction of the image slices,
            defaults to `0`
        :type axis: int, optional
        :param interval: Delay between frames in milliseconds,
            defaults to `1000`
        :type interval: int, optional
        :param blit: Whether blitting is used to optimize drawing,
            default to `True`
        :type blit: bool, optional
        :param repeat: Whether the animation repeats when the sequence
            of frames is completed, defaults to `True`
        :type repeat: bool, optional
        :param repeat_delay: Delay in milliseconds between consecutive
            animation runs if repeat is `True`, defaults to `1000`
        :type repeat_delay: int, optional
        :return: The matplotlib animation.
        :rtype: matplotlib.animation.ArtistAnimation
        """
        # Third party modules
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        # Get the frames
        data = self.unwrap_pipelinedata(data)[-1]
        delta = int(data.shape[axis]/(num_frames+1))
        indices = np.linspace(delta, data.shape[axis]-delta, num_frames)
        if data.ndim == 3:
            if not axis:
                frames = [data[int(index)] for index in indices]
            elif axis == 1:
                frames = [data[:,int(index),:] for index in indices]
            elif axis == 2:
                frames = [data[:,:,int(index)] for index in indices]
        else:
            raise ValueError('Invalid data dimension (must be 2D or 3D)')

        fig = plt.figure()
#        vmin = np.min(frames)/8
#        vmax = np.max(frames)/8
        ims = [[plt.imshow(
                    #frames[n], vmin=vmin,vmax=vmax, cmap='gray',
                    frames[n], cmap='gray',
                    animated=True)]
               for n in range(num_frames)]
        ani = animation.ArtistAnimation(
            fig, ims, interval=interval, blit=blit, repeat=repeat,
            repeat_delay=repeat_delay)

        plt.show()

        return ani


class AsyncProcessor(Processor):
    """A Processor to process multiple sets of input data via asyncio
    module.

    :ivar mgr: The `Processor` used to process every set of input data.
    :type mgr: Processor
    """
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr

    def process(self, data):
        """Asynchronously process the input documents with the
        `self.mgr` `Processor`.

        :param data: Input data documents to process.
        :type docs: iterable
        """
        # System modules
        import asyncio

        async def task(mgr, doc):
            """Process given data using provided `Processor`.

            :param mgr: The object that will process given data.
            :type mgr: Processor
            :param doc: The data to process.
            :type doc: object
            :return: The processed data.
            :rtype: object
            """
            return mgr.process(doc)

        async def execute_tasks(mgr, docs):
            """Process given set of documents using provided task
            manager.

            :param mgr: The object that will process all documents.
            :type mgr: Processor
            :param docs: The set of data documents to process.
            :type doc: iterable
            """
            coroutines = [task(mgr, d) for d in docs]
            await asyncio.gather(*coroutines)

        asyncio.run(execute_tasks(self.mgr, data))


class ImageProcessor(Processor):
    """A Processor to plot an image slice from a dataset.
    """
    def process(self, data, index=0, axis=0):
        """Plot an image from a dataset contained in `data` and return
        the full dataset.

        :param data: Input data.
        :type data: CHAP.pipeline.PipelineData
        :param index: Array index of the slice of data to plot,
            defaults to `0`
        :type index: int, optional
        :param axis: Axis direction of the image slice,
            defaults to `0`
        :type axis: int, optional
        :return: The full input dataset.
        :rtype: object
        """
        # Local modules
        from CHAP.utils.general import quick_imshow

        data = self.unwrap_pipelinedata(data)[0]
        if data.ndim == 2:
            quick_imshow(data, block=True)
        elif data.ndim == 3:
            if not axis:
                quick_imshow(data[index], block=True)
            elif axis == 1:
                quick_imshow(data[:,index,:], block=True)
            elif axis == 2:
                quick_imshow(data[:,:,index], block=True)
            else:
                raise ValueError(f'Invalid parameter axis ({axis})')
        else:
            raise ValueError('Invalid data dimension (must be 2D or 3D)')

        return data


class IntegrationProcessor(Processor):
    """A processor for integrating 2D data with pyFAI.
    """
    def process(self, data):
        """Integrate the input data with the integration method and
        keyword arguments supplied in `data` and return the results.

        :param data: Input data, containing the raw data, integration
            method, and keyword args for the integration method.
        :type data: CHAP.pipeline.PipelineData
        :return: Integrated raw data.
        :rtype: pyFAI.containers.IntegrateResult
        """
        detector_data, integration_method, integration_kwargs = data

        return integration_method(detector_data, **integration_kwargs)


class IntegrateMapProcessor(Processor):
    """A processor that takes a map and integration configuration and
    returns a NeXus NXprocesss object containing a map of the
    integrated detector data requested.
    """
    def process(self, data):
        """Process the output of a `Reader` that contains a map and
        integration configuration and return a NeXus NXprocess object
        containing a map of the integrated detector data requested.

        :param data: Input data, containing at least one item
            with the value `'MapConfig'` for the `'schema'` key, and at
            least one item with the value `'IntegrationConfig'` for the
            `'schema'` key.
        :type data: CHAP.pipeline.PipelineData
        :return: Integrated data and process metadata.
        :rtype: nexusformat.nexus.NXprocess
        """
        map_config = self.get_config(
            data, 'common.models.map.MapConfig')
        integration_config = self.get_config(
            data, 'common.models.integration.IntegrationConfig')
        nxprocess = self.get_nxprocess(map_config, integration_config)

        return nxprocess

    def get_nxprocess(self, map_config, integration_config):
        """Use a `MapConfig` and `IntegrationConfig` to construct a
        NeXus NXprocess object.

        :param map_config: A valid map configuration.
        :type map_config: MapConfig
        :param integration_config: A valid integration configuration
        :type integration_config: IntegrationConfig.
        :return: The integrated detector data and metadata.
        :rtype: nexusformat.nexus.NXprocess
        """
        # System modules
        from json import dumps
        from time import time

        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXfield,
            NXprocess,
        )
        import pyFAI

        self.logger.debug('Constructing NXprocess')
        t0 = time()

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
        for map_index in np.ndindex(map_config.shape):
            scans, scan_number, scan_step_index = \
                map_config.get_scan_step_index(map_index)
            detector_data = scans.get_detector_data(
                integration_config.detectors,
                scan_number,
                scan_step_index)
            result = integration_processor.process(
                (detector_data,
                 integration_method, integration_kwargs))
            nxprocess.data.I[map_index] = result.intensity

            scanparser = scans.get_scanparser(scan_number)
            for detector in integration_config.detectors:
                nxprocess[detector.prefix].raw_data_files[map_index] =\
                    scanparser.get_detector_data_file(
                        detector.prefix, scan_step_index)

        self.logger.debug(f'Constructed NXprocess in {time()-t0:.3f} seconds')

        return nxprocess


class MapProcessor(Processor):
    """A Processor that takes a map configuration and returns a NeXus
    NXentry object representing that map's metadata and any
    scalar-valued raw data requested by the supplied map configuration.
    """
    def process(self, data):
        """Process the output of a `Reader` that contains a map
        configuration and returns a NeXus NXentry object representing
        the map.

        :param data: Result of `Reader.read` where at least one item
            has the value `'MapConfig'` for the `'schema'` key.
        :type data: CHAP.pipeline.PipelineData
        :return: Map data and metadata.
        :rtype: nexusformat.nexus.NXentry
        """
        map_config = self.get_config(data, 'common.models.map.MapConfig')
        nxentry = self.__class__.get_nxentry(map_config)

        return nxentry

    @staticmethod
    def get_nxentry(map_config):
        """Use a `MapConfig` to construct a NeXus NXentry object.

        :param map_config: A valid map configuration.
        :type map_config: MapConfig
        :return: The map's data and metadata contained in a NeXus
            structure.
        :rtype: nexusformat.nexus.NXentry
        """
        # System modules
        from json import dumps

        # Third party modules
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXfield,
            NXsample,
        )

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
        if map_config.map_type == 'structured':
            nxentry.data.attrs['axes'] = map_config.dims
        for i, dim in enumerate(map_config.independent_dimensions[::-1]):
            nxentry.data[dim.label] = NXfield(
                value=map_config.coords[dim.label],
                units=dim.units,
                attrs={'long_name': f'{dim.label} ({dim.units})',
                       'data_type': dim.data_type,
                       'local_name': dim.name})
            if map_config.map_type == 'structured':
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

        for data in map_config.all_scalar_data:
            for map_index in np.ndindex(map_config.shape):
                nxentry.data[data.label][map_index] = map_config.get_value(
                    data, map_index)

        return nxentry


class NexusToNumpyProcessor(Processor):
    """A Processor to convert the default plottable data in a NeXus
    object into a `numpy.ndarray`.
    """
    def process(self, data):
        """Return the default plottable data signal in a NeXus object 
        contained in `data` as an `numpy.ndarray`.

        :param data: Input data.
        :type data: nexusformat.nexus.NXobject
        :raises ValueError: If `data` has no default plottable data
            signal.
        :return: The default plottable data signal.
        :rtype: numpy.ndarray
        """
        # Third party modules
        from nexusformat.nexus import NXdata

        data = self.unwrap_pipelinedata(data)[-1]

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
    """A Processor to convert the default plottable data in a
    NeXus object into an `xarray.DataArray`.
    """
    def process(self, data):
        """Return the default plottable data signal in a NeXus object
        contained in `data` as an `xarray.DataArray`.

        :param data: Input data.
        :type data: nexusformat.nexus.NXobject
        :raises ValueError: If metadata for `xarray` is absent from
            `data`
        :return: The default plottable data signal.
        :rtype: xarray.DataArray
        """
        # Third party modules
        from nexusformat.nexus import NXdata
        from xarray import DataArray

        data = self.unwrap_pipelinedata(data)[-1]

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

        :param data: Input data.
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


class RawDetectorDataMapProcessor(Processor):
    """A Processor to return a map of raw derector data in a
    NeXus NXroot object.
    """
    def process(self, data, detector_name, detector_shape):
        """Process configurations for a map and return the raw
        detector data data collected over the map.

        :param data: Input map configuration.
        :type data: CHAP.pipeline.PipelineData
        :param detector_name: The detector prefix.
        :type detector_name: str
        :param detector_shape: The shape of detector data for a single
            scan step.
        :type detector_shape: list
        :return: Map of raw detector data.
        :rtype: nexusformat.nexus.NXroot
        """
        map_config = self.get_config(data)
        nxroot = self.get_nxroot(map_config, detector_name, detector_shape)

        return nxroot

    def get_config(self, data):
        """Get instances of the map configuration object needed by this
        `Processor`.

        :param data: Result of `Reader.read` where at least one item
            has the value `'MapConfig'` for the `'schema'` key.
        :type data: CHAP.pipeline.PipelineData
        :raises Exception: If a valid map config object cannot be
            constructed from `data`.
        :return: A valid instance of the map configuration object with
            field values taken from `data`.
        :rtype: MapConfig
        """
        # Local modules
        from CHAP.common.models.map import MapConfig

        map_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if schema == 'MapConfig':
                        map_config = item.get('data')

        if not map_config:
            raise ValueError('No map configuration found in input data')

        return MapConfig(**map_config)

    def get_nxroot(self, map_config, detector_name, detector_shape):
        """Get a map of the detector data collected by the scans in
        `map_config`. The data will be returned along with some
        relevant metadata in the form of a NeXus structure.

        :param map_config: The map configuration.
        :type map_config: MapConfig
        :param detector_name: The detector prefix.
        :type detector_name: str
        :param detector_shape: The shape of detector data for a single
            scan step.
        :type detector_shape: list
        :return: A map of the raw detector data.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXinstrument,
            NXroot,
        )

        # Local modules
        from CHAP.common import MapProcessor

        nxroot = NXroot()

        nxroot[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxentry = nxroot[map_config.title]

        nxentry.instrument = NXinstrument()
        nxentry.instrument.detector = NXdetector()

        nxentry.instrument.detector.data = NXdata()
        nxdata = nxentry.instrument.detector.data
        nxdata.raw = np.empty((*map_config.shape, *detector_shape))
        nxdata.raw.attrs['units'] = 'counts'
        for i, det_axis_size in enumerate(detector_shape):
            nxdata[f'detector_axis_{i}_index'] = np.arange(det_axis_size)

        for map_index in np.ndindex(map_config.shape):
            scans, scan_number, scan_step_index = \
                map_config.get_scan_step_index(map_index)
            scanparser = scans.get_scanparser(scan_number)
            self.logger.debug(
                f'Adding data to nxroot for map point {map_index}')
            nxdata.raw[map_index] = scanparser.get_detector_data(
                detector_name,
                scan_step_index)

        nxentry.data.makelink(
            nxdata.raw,
            name=detector_name)
        for i, det_axis_size in enumerate(detector_shape):
            nxentry.data.makelink(
                nxdata[f'detector_axis_{i}_index'],
                name=f'{detector_name}_axis_{i}_index'
            )
            if isinstance(nxentry.data.attrs['axes'], str):
                nxentry.data.attrs['axes'] = [
                    nxentry.data.attrs['axes'],
                    f'{detector_name}_axis_{i}_index']
            else:
                nxentry.data.attrs['axes'] += [
                    f'{detector_name}_axis_{i}_index']

        nxentry.data.attrs['signal'] = detector_name

        return nxroot


class StrainAnalysisProcessor(Processor):
    """A Processor to compute a map of sample strains by fitting Bragg
    peaks in 1D detector data and analyzing the difference between
    measured peak locations and expected peak locations for the sample
    measured.
    """
    def process(self, data):
        """Process the input map detector data & configuration for the
        strain analysis procedure, and return a map of sample strains.

        :param data: Results of `MutlipleReader.read` containing input
            map detector data and strain analysis configuration
        :type data: CHAP.pipeline.PipelineData
        :return: A map of sample strains.
        :rtype: xarray.Dataset
        """
        strain_analysis_config = self.get_config(data)

        return data

    def get_config(self, data):
        """Get instances of the configuration objects needed by this
        `Processor`.

        :param data: Result of `Reader.read` where at least one item
            has the value `'StrainAnalysisConfig'` for the `'schema'`
            key.
        :type data: CHAP.pipeline.PipelineData
        :raises Exception: If valid config objects cannot be
            constructed from `data`.
        :return: A valid instance of the configuration object with
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
    """A Processor to convert the data in an `xarray` structure to a
    NeXus NXdata object.
    """
    def process(self, data):
        """Return `data` represented as a NeXus NXdata object.

        :param data: The input `xarray` structure.
        :type data: typing.Union[xarray.DataArray, xarray.Dataset]
        :return: The data and metadata in `data`.
        :rtype: nexusformat.nexus.NXdata
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        data = self.unwrap_pipelinedata(data)[-1]
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

        :param data: The input `xarray.DataArray`.
        :type data: xarray.DataArray
        :return: The data in `data`.
        :rtype: numpy.ndarray
        """

        return self.unwrap_pipelinedata(data)[-1].data


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
