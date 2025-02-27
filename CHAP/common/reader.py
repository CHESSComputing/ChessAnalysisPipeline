#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific
             workflows.
"""

# System modules
from os.path import (
    isabs,
    isfile,
    join,
    splitext,
)
from sys import modules

# Third party modules
import numpy as np

# Local modules
from CHAP import Reader


class BinaryFileReader(Reader):
    """Reader for binary files."""
    def read(self, filename):
        """Return a content of a given binary file.

        :param filename: The name of the binary file to read from.
        :type filename: str
        :return: The file content.
        :rtype: binary
        """
        with open(filename, 'rb') as file:
            data = file.read()
        return data


class FabioImageReader(Reader):
    """Reader for images using the python package
    [`fabio`](https://fabio.readthedocs.io/en/main/).
    """
    def read(self, filename, frame=None):
        """Return the data from the image file(s) provided.

        :param filename: The image filename, or glob pattern for image
            filenames, to read.
        :type filename: str
        :param frame: The index of a specific frame to read from the
            file(s), defaults to `None`.
        :type filename: int, optional
        :returns: Image data as a numpy array (or list of numpy
            arrays, if a glob pattern matching more than one file was
            provided).
        """
        from glob import glob
        import fabio

        filenames = glob(filename)
        data = []
        for f in filenames:
            image = fabio.open(f, frame=frame)
            data.append(image.data)
            image.close()
        return data


class H5Reader(Reader):
    """Reader for h5 files."""
    def read(self, filename, h5path='/', idx=None):
        """Return the data object stored at `h5path` in an h5-file.

        :param filename: The name of the h5-file to read from.
        :type filename: str
        :param h5path: The path to a specific location in the h5 file
            to read data from, defaults to `'/'`.
        :type h5path: str, optional
        :return: The object indicated by `filename` and `h5path`.
        :rtype: object
        """
        # Third party modules
        from h5py import File

        data = File(filename, 'r')[h5path]
        if idx is not None:
            data = data[tuple(idx)]
        return data


class LinkamReader(Reader):
    """Reader for loading Linkam load frame .txt files as an
    `NXdata`.
    """
    def read(self, filename, columns=None, inputdir=None):
        """Read specified columns from the given Linkam file.

        :param filename: Name of Linkam .txt file
        :type filename: str
        :param columns: Column names to read in, defaults to None
            (read in all columns)
        :type columns: list[str], optional
        :returns: Linkam data represented in an `NXdata` object
        :rtype: nexusformat.nexus.NXdata
        """
        from nexusformat.nexus import NXdata, NXfield

        # Parse .txt file
        start_time, metadata, data = self.__class__.parse_file(
            filename, self.logger)

        # Get list of actual data column names and corresponding
        # signal nxnames (same as user-supplied column names)
        signal_names = []
        if columns is None:
            signal_names = [(col, col) for col in data.keys() if col != 'Time']
        else:
            for col in columns:
                col_actual = col
                if not col in data:
                    if f'{col}_Y' in data:
                        # Always use the *_Y column if the user-supplied
                        # column name has both _X and _Y components
                        col_actual = f'{col}_Y'
                    else:
                        self.logger.warning(f'{col} not present in {filename}')
                        continue
                signal_names.append((col_actual, col))
        self.logger.info(f'Using (column name, signal name): {signal_names}')

        nxdata = NXdata(
            axes=(NXfield(
                name='Time',
                value=np.array(data['Time']) + start_time,
                dtype='float64',
            ),),
            **{col: NXfield(
                name=col,
                value=data[col_actual],
                dtype='float32',
            ) for col, col_actual in signal_names},
            attrs=metadata
        )
        return nxdata

    @classmethod
    def parse_file(cls, filename, logger):
        """Return start time, metadata, and data stored in the
        provided Linkam .txt file.

        :param filename: Name of Linkam .txt file
        :type filename: str
        :returns:
        :rtype: tuple(float, dict[str, str], dict[str, list[float]])
        """
        from datetime import datetime
        import os
        import re

        # Get t=0 from filename
        start_time = None
        basename = os.path.basename(filename)
        pattern = r'(\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{2})'
        match = re.search(pattern, basename)
        if match:
            datetime_str = match.group(1)
            dt = datetime.strptime(datetime_str, '%y-%m-%d_%H-%M-%S-%f')
            start_time = dt.timestamp()
        else:
            logger.warning('Datetime not found in filename')

        # Get data add metadata from file contents
        metadata = {}
        data = False
        with open(filename, 'r', encoding='utf-8') as inf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                if data:
                    # If data dict has been initialized, remaining
                    # lines are all data values
                    values = line.split('\t')
                    for val, col in zip(values, list(data.keys())):
                        try:
                            val = float(val)
                        except:
                            logger.error(
                                f'Cannot convert {col} value to float: {val}')
                            continue
                        else:
                            data[col].append(val)
                if ':' in line:
                    # Metadata key: value pair kept on this line
                    _metadata = line.split(':', 1)
                    if len(_metadata) == 2:
                        key, val = _metadata
                    else:
                        continue
                        key, val = _metadata[0], None
                    metadata[key] = val
                if re.match(r'^Temperature', line):
                    # Match found for start of data section -- this
                    # line and the next are column labels.
                    data_cols = []
                    # Get base quantity column names
                    base_cols = line.replace(
                        'Force V Distance', 'Force\t\tDistance').split('\t\t')
                    # Get Index, X and Y component columns
                    line = next(inf)
                    comp_cols = line.split('\t')
                    # Assemble actual column names
                    data_cols.append('Index')
                    comp_cols_count = int((len(comp_cols) - 1) / 2)
                    for i in range(comp_cols_count):
                        data_cols.extend(
                            [f'{base_cols[i]}_{comp}' for comp in ('X', 'Y')]
                        )
                    if len(base_cols) > comp_cols_count:
                        data_cols.extend(base_cols[comp_cols_count - 1:])
                    # Temperature_X is actually Time (?)
                    data_cols[data_cols.index('Temperature_X')] = 'Time'
                    # Start of data lines
                    data = {col: [] for col in data_cols}
                    logger.info(f'Found data columns: {data_cols}')

        return start_time, metadata, data



class MapReader(Reader):
    """Reader for CHESS sample maps."""
    def read(
            self, filename=None, map_config=None, detector_names=None,
            inputdir=None):
        """Take a map configuration dictionary and return a
        representation of the map as a NeXus NXentry object. The
        NXentry's default data group will contain the raw data
        collected over the course of the map.

        :param filename: The name of a file with the map configuration
            to read and pass onto the constructor of
            `CHAP.common.models.map.MapConfig`.
        :type filename: str, optional
        :param map_config: A map configuration to be passed directly to
            the constructor of `CHAP.common.models.map.MapConfig`.
        :type map_config: dict, optional
        :param detector_names: Detector prefixes to include raw data
            for in the returned NeXus NXentry object.
        :type detector_names: list[str], optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :return: Data from the provided map configuration.
        :rtype: nexusformat.nexus.NXentry
        """
        # Third party modules
        from json import dumps
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXfield,
            NXsample,
        )

        # Local modules
        from CHAP.common.models.map import MapConfig

        raise RuntimeError('MapReader is obsolete, use MapProcessor')

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
        nxentry[map_config.sample.name] = NXsample(
            **map_config.sample.dict())

        # Set up default data group
        nxentry.data = NXdata()
        if map_config.map_type == 'structured':
            nxentry.data.attrs['axes'] = map_config.dims
        for dim in map_config.independent_dimensions:
            nxentry.data[dim.label] = NXfield(
                value=map_config.coords[dim.label],
                units=dim.units,
                attrs={'long_name': f'{dim.label} ({dim.units})',
                       'data_type': dim.data_type,
                       'local_name': dim.name})

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
        if detector_names is None:
            detector_names = []
        for detector_name in detector_names:
            if not isinstance(detector_name, str):
                detector_name = str(detector_name)
            detector_data = map_config.get_detector_data(
                detector_name, (0,) * len(map_config.shape))
            nxentry.data[detector_name] = NXfield(value=np.zeros(
                (*map_config.shape, *detector_data.shape)),
                dtype=detector_data.dtype)

        # Read and fill in maps of raw data
        if len(map_config.all_scalar_data) > 0 or detector_names:
            for map_index in np.ndindex(map_config.shape):
                for data in map_config.all_scalar_data:
                    nxentry.data[data.label][map_index] = map_config.get_value(
                        data, map_index)
                for detector_name in detector_names:
                    if not isinstance(detector_name, str):
                        detector_name = str(detector_name)
                    nxentry.data[detector_name][map_index] = \
                        map_config.get_detector_data(detector_name, map_index)

        return nxentry


class NexusReader(Reader):
    """Reader for NeXus files."""
    def read(self, filename, nxpath='/', nxmemory=2000):
        """Return the NeXus object stored at `nxpath` in a NeXus file.

        :param filename: The name of the NeXus file to read from.
        :type filename: str
        :param nxpath: The path to a specific location in the NeXus
            file tree to read from, defaults to `'/'`.
        :type nxpath: str, optional
        :raises nexusformat.nexus.NeXusError: If `filename` is not a
            NeXus file or `nxpath` is not in its tree.
        :return: The NeXus object indicated by `filename` and `nxpath`.
        :rtype: nexusformat.nexus.NXobject
        """
        # Third party modules
        from nexusformat.nexus import nxload
        from nexusformat.nexus.tree import NX_CONFIG
        NX_CONFIG['memory'] = nxmemory

        return nxload(filename)[nxpath]


class NXdataReader(Reader):
    """Reader for constructing an NXdata object from components."""
    def read(
            self, name, nxfield_params, signal_name, axes_names, attrs=None,
            inputdir='.'):
        """Return a basic NXdata object constructed from components.

        :param name: The name of the NXdata group.
        :type name: str
        :param nxfield_params: List of sets of parameters for
            `NXfieldReader` specifying the NXfields belonging to the
            NXdata.
        :type nxfield_params: list[dict]
        :param signal_name: Name of the signal for the NXdata (must be
            one of the names of the NXfields indicated in `nxfields`).
        :type signal: str
        :param axes_names: Name or names of the coordinate axes
            NXfields associated with the signal (must be names of
            NXfields indicated in `nxfields`).
        :type axes_names: Union[str, list[str]]
        :param attrs: Dictionary of additional attributes for the
            NXdata.
        :type attrs: dict, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str
        :returns: A new NXdata object.
        :rtype: nexusformat.nexus.NXdata
        """
        from nexusformat.nexus import NXdata

        # Read in NXfields
        nxfields = [NXfieldReader().read(**params, inputdir=inputdir)
                    for params in nxfield_params]
        nxfields = {nxfield.nxname: nxfield for nxfield in nxfields}

        # Get signal NXfield
        try:
            nxsignal = nxfields[signal_name]
        except Exception as exc:
            raise ValueError(
                '`signal_name` must be the name of one of the NXfields '
                'indicated in `nxfields`: , '.join(nxfields.keys())) from exc

        # Get axes NXfield(s)
        if isinstance(axes_names, str):
            axes_names = [axes_names]
        try:
            nxaxes = [nxfields[axis_name] for axis_name in axes_names]
        except Exception as exc:
            raise ValueError(
                '`axes_names` must contain only names of NXfields indicated '
                'in `nxfields`: ' + ', '.join(nxfields.keys())) from exc
        for i, nxaxis in enumerate(nxaxes):
            if len(nxaxis) != nxsignal.shape[i]:
                raise ValueError(
                    f'Shape mismatch on signal dimension {i}: signal '
                    + f'"{nxsignal.nxname}" has {nxsignal.shape[i]} values, '
                    + f'but axis "{nxaxis.nxname}" has {len(nxaxis)} values.')

        if attrs is None:
            attrs = {}
        result = NXdata(signal=nxsignal, axes=nxaxes, name=name, attrs=attrs,
                        **nxfields)
        self.logger.info(result.tree)
        return result


class NXfieldReader(Reader):
    """Reader for an NXfield with options to modify certain attributes.
    """
    def read(
            self, filename, nxpath, nxname=None, update_attrs=None,
            slice_params=None, inputdir='.'):
        """Return a copy of the indicated NXfield from the file. Name
        and attributes of the returned copy may be modified with the
        `nxname` and `update_attrs` keyword arguments.

        :param filename: Name of the NeXus file containing the NXfield
            to read.
        :type filename: str
        :param nxpath: Path in `nxfile` pointing to the NXfield to
           read.
        :type nxpath: str
        :param nxname: New name for the returned NXfield.
        :type nxname: str, optional
        :param update_attrs: Optional dictonary used to add to /
            update the original NXfield's attributes.
        :type update_attrs: dict, optional
        :param slice_params: Parameters for returning just a slice of
            the full field data. Slice parameters are provided in a
            list dictionaries with integer values for any / all of the
            following keys: `"start"`, `"end"`, `"step"`. Default
            values used are: `"start"` - `0`, `"end"` -- `None`,
            `"step"` -- `1`. The order of the list must correspond to
            the order of the field's axes.
        :type slice_params: list[dict[str, int]], optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str
        :returns: A copy of the indicated NXfield (with name and
            attributes optionally modified).
        :rtype: nexusformat.nexus.NXfield
        """
        # Third party modules
        from nexusformat.nexus import (
            NXfield,
            nxload,
        )

        if not isabs(filename):
            filename = join(inputdir, filename)
        nxroot = nxload(filename)
        nxfield = nxroot[nxpath]

        if nxname is None:
            nxname = nxfield.nxname

        attrs = nxfield.attrs
        if update_attrs is not None:
            attrs.update(update_attrs)

        if slice_params is None:
            value = nxfield.nxdata
        else:
            if len(slice_params) < nxfield.ndim:
                slice_params.extend([{}] * (nxfield.ndim - len(slice_params)))
            if len(slice_params) > nxfield.ndim:
                slice_params = slice_params[0:nxfield.ndim]
            slices = ()
            default_slice = {'start': 0, 'end': None, 'step': 1}
            for s in slice_params:
                for k, v in default_slice.items():
                    if k not in s:
                        s[k] = v
                slices = (*slices, slice(s['start'], s['end'], s['step']))
            value = nxfield.nxdata[slices]

        nxfield = NXfield(value=value, name=nxname, attrs=attrs)
        self.logger.debug(f'Result -- nxfield.tree =\n{nxfield.tree}')

        return nxfield


class SpecReader(Reader):
    """Reader for CHESS SPEC scans."""
    def read(
            self, filename=None, config=None, detectors=None,
            inputdir=None):
        """Take a SPEC configuration filename or dictionary and return
        the raw data as a Nexus NXentry object.

        :param filename: The name of file with the SPEC configuration
            to read from to pass onto the constructor of
            `CHAP.common.models.map.SpecConfig`.
        :type filename: str, optional
        :param config: A SPEC configuration to be passed directly
            to the constructor of `CHAP.common.models.map.SpecConfig`.
        :type config: dict, optional
        :param detectors: Detector configurations of the detectors
            to include raw data for in the returned NeXus output,
            defaults to None (only a valid input for EDD).
        :type detectors: CHAP.common.models.map.DetectorConfig,
            optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str
        :return: The data from the provided SPEC configuration.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from json import dumps
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXfield,
            NXroot,
        )

        # Local modules
        from CHAP.common.models.map import (
            Detector,
            DetectorConfig,
            SpecConfig,
        )

        if filename is not None:
            if config is not None:
                raise RuntimeError('Specify either filename or config '
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
            config = reader.read(filename)
        elif not isinstance(config, dict):
            raise RuntimeError('Invalid parameter config in '
                               f'common.SpecReader ({config})')

        # Validate the SPEC configuration provided by constructing a
        # SpecConfig
        config = SpecConfig(**config, inputdir=inputdir)

        # Validate the detector configuration
        if detectors is None:
            if config.experiment_type != 'EDD':
                raise RuntimeError('Missing parameter detectors for '
                                   f'experiment type {config.experiment_type}')
        else:
            detectors = DetectorConfig(detectors=detectors)

        # Create the NXroot object
        nxroot = NXroot()
        nxentry = NXentry(name=config.experiment_type)
        nxroot[nxentry.nxname] = nxentry
        nxentry.set_default()

        # Set up NXentry and add misc. CHESS-specific metadata as well
        # as all spec_motors, scan_columns, and smb_pars, and the
        # detector info and raw detector data
        nxentry.config = dumps(config.dict())
        nxentry.attrs['station'] = config.station
        if config.experiment_type == 'EDD':
            if detectors is None:
                detectors_ids = None
            else:
                try:
                    detectors_ids = [int(d.id) for d in detectors.detectors]
                except:
                    detectors_ids = [d.id for d in detectors.detectors]
        nxentry.spec_scans = NXcollection()
#        nxpaths = []
        if config.experiment_type == 'EDD':
            detector_data_format = None
        for scans in config.spec_scans:
            nxscans = NXcollection()
            nxentry.spec_scans[f'{scans.scanparsers[0].scan_name}'] = nxscans
            nxscans.attrs['spec_file'] = str(scans.spec_file)
            nxscans.attrs['scan_numbers'] = scans.scan_numbers
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                if config.experiment_type == 'EDD':
                    if detector_data_format is None:
                        detector_data_format = scanparser.detector_data_format
                    elif (scanparser.detector_data_format !=
                            detector_data_format):
                        raise ValueError(
                        'Mixing `spec` and `h5` data formats not implemented')
                nxscans[scan_number] = NXcollection()
                try:
                    nxscans[scan_number].spec_motors = dumps(
                        {k:float(v) for k,v
                         in scanparser.spec_positioner_values.items()})
                except:
                    pass
                try:
                    nxscans[scan_number].scan_columns = dumps(
                        {k:list(v) for k,v
                         in scanparser.spec_scan_data.items() if len(v)})
                except:
                    pass
                try:
                    nxscans[scan_number].smb_pars = dumps(
                        {k:v for k,v in scanparser.pars.items()})
                except:
                    pass
                try:
                    nxscans[scan_number].spec_scan_motor_mnes = dumps(
                        scanparser.spec_scan_motor_mnes)
                except:
                    pass
                if config.experiment_type == 'EDD':
                    nxdata = NXdata()
                    nxscans[scan_number].data = nxdata
#                    nxpaths.append(
#                        f'spec_scans/{nxscans.nxname}/{scan_number}/data')
                    nxdata.data = NXfield(
                        value=scanparser.get_detector_data(detectors_ids)[0])
                else:
                    nxdata = NXdata()
                    nxscans[scan_number].data = nxdata
#                    nxpaths.append(
#                        f'spec_scans/{nxscans.nxname}/{scan_number}/data')
                    for detector in detectors.detectors:
                        nxdata[detector.id] = NXfield(
                           value=scanparser.get_detector_data(detector.id))

        if detectors is None and config.experiment_type == 'EDD':
            if detector_data_format == 'spec':
                detectors = DetectorConfig(
                    detectors=[Detector(id='mca1')
                               for i in range(nxdata.data.shape[1])])
            else:
                detectors = DetectorConfig(
                    detectors=[
                        Detector(id=i) for i in range(nxdata.data.shape[1])])
        nxentry.detectors = dumps(detectors.dict())

        #return nxroot, nxpaths
        return nxroot


class URLReader(Reader):
    """Reader for data available over HTTPS."""
    def read(self, url, headers=None, timeout=10):
        """Make an HTTPS request to the provided URL and return the
        results. Headers for the request are optional.

        :param url: The URL to read.
        :type url: str
        :param headers: Headers to attach to the request.
        :type headers: dict, optional
        :param timeout: Timeout for the HTTPS request,
            defaults to `10`.
        :type timeout: int
        :return: The content of the response.
        :rtype: object
        """
        # System modules
        import requests

        if headers is None:
            headers = {}
        resp = requests.get(url, headers=headers, timeout=timeout)
        data = resp.content

        self.logger.debug(f'Response content: {data}')

        return data


class YAMLReader(Reader):
    """Reader for YAML files."""
    def read(self, filename):
        """Return a dictionary from the contents of a yaml file.

        :param filename: The name of the YAML file to read from.
        :type filename: str
        :return: The contents of the file.
        :rtype: dict
        """
        # Third party modules
        import yaml

        with open(filename) as file:
            data = yaml.safe_load(file)
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.reader import main

    main()
