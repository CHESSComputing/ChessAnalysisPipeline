#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific
             workflows.
"""

# system modules
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


class MapReader(Reader):
    """Reader for CHESS sample maps"""
    def read(self, map_config):
        """Take a map configuration dictionary and return a
        representation of the map as an NXentry. The NXentry's default
        data group will contain the raw data collected over the course
        of the map.

        :param map_config: map configurationto be passed directly to
            the constructor of `CHAP.common.models.map.MapConfig`
        :type map_config: dict
        :return: Data from the map configuration provided
        :rtype: nexusformat.nexus.NXentry
        """
        from json import dumps
        from nexusformat.nexus import (NXcollection,
                                       NXdata,
                                       NXentry,
                                       NXfield,
                                       NXsample)
        import numpy as np
        from CHAP.common.models.map import MapConfig

        # Validate the map configuration provided by constructing a
        # MapConfig
        map_config = MapConfig(**map_config)

        # Set up NXentry and add misc. CHESS-specific metadata
        nxentry = NXentry(name=map_config.title)
        nxentry.map_config = dumps(map_config.dict())

        nxentry.attrs['station'] = map_config.station
        nxentry.spec_scans = NXcollection()
        for scans in map_config.spec_scans:
            nxentry.spec_scans[scans.scanparsers[0].scan_name] = \
                NXfield(value=scans.scan_numbers,
                        dtype='int8',
                        attrs={'spec_file': str(scans.spec_file)})

        # Add sample metadata
        nxentry[map_config.sample.name] = NXsample(**map_config.sample.dict())

        # Set up default data group
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

        # Set up empty NXfields for scalar data present in the map
        # configuration provided
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

        # Fill in the empty NXfields with maps of raw scan data
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
