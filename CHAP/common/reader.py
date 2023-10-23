#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific
             workflows.
"""

# system modules
from os.path import (
    isfile,
    splitext,
)
from sys import modules
from time import time

# local modules
from CHAP import Reader


class BinaryFileReader(Reader):
    """Reader for binary files"""
    def read(self, filename):
        """Return a content of a given file name

        :param filename: name of the binart file to read from
        :return: the content of `filename`
        :rtype: binary
        """

        with open(filename, 'rb') as file:
            data = file.read()
        return data


class SpecReader(Reader):
    """Reader for CHESS SPEC scans"""
    def read(self, filename=None, spec_config=None, detector_names=[],
            inputdir=None):
        """Take a SPEC configuration filename or dictionary and return
        the raw data as an NXentry.

        :param filename: name of file with the SPEC configuration to
            read and pass onto the constructor of
            `CHAP.common.models.map.SpecConfig`
        :type filename: str, optional
        :param spec_config: SPEC configuration to be passed directly
            to the constructor of
            `CHAP.common.models.map.SpecConfig`
        :type spec_config: dict, optional
        :param detector_names: Detector prefixes to include raw data
            for in the returned NXentry
        :type detector_names: list[str]
        :return: Data from the SPEC configuration provided
        :rtype: nexusformat.nexus.NXentry
        """
        from json import dumps
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXfield,
        )
        import numpy as np
        from CHAP.common.models.map import SpecConfig

        if filename is not None:
            if spec_config is not None:
                raise RuntimeError('Specify either filename or spec_config '
                                   'in common.SpecReader, not both')
            # Read the map configuration from file
            if not isfile(filename):
                raise OSError(f'input file does not exist ({filename})')
            extension = splitext(filename)[1]
            if extension in ('.yml', '.yaml'):
                reader = YAMLReader()
            else:
                raise RuntimeError('input file has a non-implemented '
                                   f'extension ({filename})')
            spec_config = reader.read(filename)
        elif not isinstance(spec_config, dict):
            raise RuntimeError('Invalid parameter spec_config in '
                               f'common.SpecReader ({spec_config})')

        # Validate the SPEC configuration provided by constructing a
        # SpecConfig
        spec_config = SpecConfig(**spec_config, inputdir=inputdir)

        # Set up NXentry and add misc. CHESS-specific metadata
        # as well as all spec_motors, scan_columns, and smb_pars
        nxentry = NXentry(name=spec_config.experiment_type)
        nxentry.spec_config = dumps(spec_config.dict())
        nxentry.attrs['station'] = spec_config.station
        nxentry.spec_scans = NXcollection()
        for scans in spec_config.spec_scans:
            nxscans = NXcollection()
            nxentry.spec_scans[f'{scans.scanparsers[0].scan_name}'] = nxscans
            nxscans.attrs['spec_file'] = str(scans.spec_file)
            nxscans.attrs['scan_numbers'] = scans.scan_numbers
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                nxscans[scan_number] = NXcollection()
                if hasattr(scanparser, 'spec_positioner_values'):
                    nxscans[scan_number].spec_motors = dumps(
                        {k:float(v) for k,v
                         in scanparser.spec_positioner_values.items()})
                if hasattr(scanparser, 'spec_scan_data'):
                    nxscans[scan_number].scan_columns = dumps(
                        {k:list(v) for k,v
                         in scanparser.spec_scan_data.items() if len(v)})
                if hasattr(scanparser, 'pars'):
                    nxscans[scan_number].smb_pars = dumps(
                        {k:v for k,v in scanparser.pars.items()})
                if detector_names:
                    nxdata = NXdata()
                    nxscans[scan_number].data = nxdata
                    for detector_name in detector_names:
                        nxdata[detector_name] = NXfield(
                           value=scanparser.get_detector_data(detector_name))

        return nxentry


class MapReader(Reader):
    """Reader for CHESS sample maps"""
    def read(
            self, filename=None, map_config=None, detector_names=[],
            inputdir=None):
        """Take a map configuration dictionary and return a
        representation of the map as an NXentry. The NXentry's default
        data group will contain the raw data collected over the course
        of the map.

        :param filename: name of file with the map configuration to
            read and pass onto the constructor of
            `CHAP.common.models.map.MapConfig`
        :type filename: str, optional
        :param map_config: map configuration to be passed directly to
            the constructor of `CHAP.common.models.map.MapConfig`
        :type map_config: dict, optional
        :param detector_names: Detector prefixes to include raw data
            for in the returned NXentry
        :type detector_names: list[str]
        :return: Data from the map configuration provided
        :rtype: nexusformat.nexus.NXentry
        """
        from json import dumps
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXfield,
            NXsample,
        )
        import numpy as np
        from CHAP.common.models.map import MapConfig

        if filename is not None:
            if map_config is not None:
                raise RuntimeError('Specify either filename or map_config '
                                   'in common.MapReader, not both')
            # Read the map configuration from file
            if not isfile(filename):
                raise OSError(f'input file does not exist ({filename})')
            extension = splitext(filename)[1]
            if extension in ('.yml', '.yaml'):
                reader = YAMLReader()
            else:
                raise RuntimeError('input file has a non-implemented '
                                   f'extension ({filename})')
            map_config = reader.read(filename)
        elif not isinstance(map_config, dict):
            raise RuntimeError('Invalid parameter map_config in '
                               f'common.MapReader ({map_config})')

        # Validate the map configuration provided by constructing a
        # MapConfig
        map_config = MapConfig(**map_config, inputdir=inputdir)

        # Set up NXentry and add misc. CHESS-specific metadata
        nxentry = NXentry(name=map_config.title)
        nxentry.attrs['station'] = map_config.station
        nxentry.map_config = dumps(map_config.dict())
        nxentry.spec_scans = NXcollection()
        for scans in map_config.spec_scans:
            nxentry.spec_scans[scans.scanparsers[0].scan_name] = \
                NXfield(value=scans.scan_numbers,
                        attrs={'spec_file': str(scans.spec_file)})

        # Add sample metadata
        nxentry[map_config.sample.name] = NXsample(**map_config.sample.dict())

        # Set up default data group
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

        # Create empty NXfields for all scalar data present in the
        # provided map configuration
        signal = False
        auxilliary_signals = []
        for data in map_config.all_scalar_data:
            nxentry.data[data.label] = NXfield(
                value=np.zeros(map_config.shape),
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

        # Create empty NXfields of appropriate shape for raw
        # detector data
        for detector_name in detector_names:
            detector_data = map_config.get_detector_data(
                detector_name, (0,) * len(map_config.shape))
            nxentry.data[detector_name] = NXfield(value=np.zeros(
                (*map_config.shape, *detector_data.shape)),
                dtype=detector_data.dtype)
#            data_shape = list(map_config.shape)+list(detector_data.shape)
#            nxentry.data[detector_name] = NXfield(
#                value=np.zeros(data_shape), shape=data_shape,
#                dtype=detector_data.dtype)

        # Read and fill in maps of raw data
        if len(map_config.all_scalar_data) > 0 or len(detector_names) > 0:
            for map_index in np.ndindex(map_config.shape):
                for data in map_config.all_scalar_data:
                    nxentry.data[data.label][map_index] = map_config.get_value(
                        data, map_index)
                for detector_name in detector_names:
                    nxentry.data[detector_name][map_index] = \
                        map_config.get_detector_data(detector_name, map_index)

        return nxentry


class NexusReader(Reader):
    """Reader for NeXus files"""
    def read(self, filename, nxpath='/'):
        """Return the NeXus object stored at `nxpath` in the nexus
        file `filename`.

        :param filename: name of the NeXus file to read from
        :type filename: str
        :param nxpath: path to a specific loaction in the NeXus file
            to read from, defaults to `'/'`
        :type nxpath: str, optional
        :raises nexusformat.nexus.NeXusError: if `filename` is not a
            NeXus file or `nxpath` is not in `filename`.
        :return: the NeXus structure indicated by `filename` and `nxpath`.
        :rtype: nexusformat.nexus.NXobject
        """

        from nexusformat.nexus import nxload

        nxobject = nxload(filename)[nxpath]
        return nxobject


class URLReader(Reader):
    """Reader for data available over HTTPS"""
    def read(self, url, headers={}, timeout=10):
        """Make an HTTPS request to the provided URL and return the
        results.  Headers for the request are optional.

        :param url: the URL to read
        :type url: str
        :param headers: headers to attach to the request, defaults to
            `{}`
        :type headers: dict, optional
        :return: the content of the response
        :rtype: object
        """

        import requests

        resp = requests.get(url, headers=headers, timeout=timeout)
        data = resp.content

        self.logger.debug(f'Response content: {data}')

        return data


class YAMLReader(Reader):
    """Reader for YAML files"""
    def read(self, filename):
        """Return a dictionary from the contents of a yaml file.

        :param filename: name of the YAML file to read from
        :return: the contents of `filename`
        :rtype: dict
        """

        import yaml

        with open(filename) as file:
            data = yaml.safe_load(file)
        return data


if __name__ == '__main__':
    from CHAP.reader import main
    main()
