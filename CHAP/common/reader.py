#!/usr/bin/env python
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Readers used in multiple experiment-specific workflows.
"""

# System modules
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
import numpy as np
from pydantic import (
    PrivateAttr,
    conint,
    conlist,
    constr,
    model_validator,
)

# Local modules
from CHAP import Reader
from CHAP.reader import validate_reader_model
from CHAP.common.models.map import (
    DetectorConfig,
    SpecConfig,
)

def validate_model(model):
    if model.filename is not None:
        validate_reader_model(model)
    return model


class BinaryFileReader(Reader):
    """Reader for binary files."""
    def read(self):
        """Return a content of a given binary file.

        :return: File content.
        :rtype: binary
        """
        with open(self.filename, 'rb') as file:
            data = file.read()
        return data


class ConfigReader(Reader):
    """Reader for YAML files that optionally implements and verifies it
    agaist its Pydantic configuration schema.
    """
    def read(self):
        """Return an optionally verified dictionary from the contents
        of a yaml file.
        """
        data = YAMLReader(**self.model_dump()).read()
        #print(f'\nConfigReader.read start data {type(data)}:')
        raise RuntimeError(
            'FIX ConfigReader downstream validators do not like a pydantic '
            'class as output of a reader, but returning data.model_dict() '
            'instead screws up default value identification')
        #pprint(data)
        if self.get_schema() is not None:
            data = self.get_config(config=data, schema=self.get_schema())
        self.status = 'read'
        #print(f'\nConfigReader.read end data {type(data)}:')
        #pprint(data)
        return data


class DetectorDataReader(Reader):
    """Reader for detector data files. Glob filenames allowed. Mask
    application and background correction available."""
    def read(self, filename,
             mask_file=None, mask_above=None, mask_below=None,
             mask_value=np.nan,
             data_scalar=None,
             background_file=None, background_scalar=None,
             ):
        """Reads detector data, applies masking, scaling, and
        background subtraction.

        :param filename: Path to the primary data file.
        :type filename: str
        :param mask_file: Path to the mask file (optional).
        :type mask_file: str, optional
        :param mask_above: Mask values above this threshold (optional).
        :type mask_above: float, optional
        :param mask_below: Mask values below this threshold (optional).
        :type mask_below: float, optional
        :param data_scalar: Scalar to multiply the data (optional).
        :type data_scalar: float, optional
        :param background_file: Path to the background file (optional).
        :type background_file: str, optional
        :param background_scalar: Scalar to multiply the background
            data (optional).
        :type background_scalar: float, optional
        :return: Processed detector data.
        :rtype: numpy.ndarray
        """
        import glob
        import os
        import numpy as np

        # Handle glob filenames
        if not os.path.isfile(filename):
            filenames = sorted(glob.glob(filename))
            if not filenames:
                raise ValueError(
                    f'{filename} is not a file or glob that matches any files')
        else:
            filenames = [filename]

        # Read the raw data files
        self.logger.info(f'Reading {len(filenames)} raw data files')
        raw_data = [self.get_data_from_file(f) for f in filenames]

        # Initialize mask arary
        self.logger.info(f'Initializing mask array')
        mask_array = np.zeros_like(raw_data[0], dtype=bool)
        if mask_file:
            mask_array |= self.get_data_from_file(mask_file) != 0

        # Initialize background data
        self.logger.info('Initializing background data')
        if background_file:
            background_data = self.get_data_from_file(background_file)
            if background_scalar:
                background_data *= background_scalar
        else:
            background_data = None

        # Handle mask_value of NaN
        if isinstance(mask_value, str):
            if mask_value.lower() == 'nan':
                mask_value = np.nan

        # Scale data, apply mask, subtract background
        self.logger.info('Applying corrections to raw data')
        corrected_data = self.correct_data(
            np.array(raw_data),
            mask_array, mask_above, mask_below, mask_value,
            data_scalar, background_data)

        return corrected_data

    def get_data_from_file(self, filename):
        import fabio

        self.logger.debug(f'Reading {filename}')
        with fabio.open(filename) as datafile:
            data = datafile.data
        return data

    def correct_data(self, raw_data,
                     mask_array, mask_above, mask_below, mask_value,
                     data_scalar,
                     background_data):
        import numpy as np

        # Scale raw data
        corrected_data = raw_data
        if data_scalar:
            corrected_data *= data_scalar

        # Apply mask
        mask = mask_array != 0
        if mask_above is not None:
            mask |= corrected_data > mask_above
        if mask_below is not None:
            mask |= corrected_data < mask_below
        corrected_data = np.where(mask, mask_value, raw_data)

        # Subtract background
        if background_data:
            corrected_data -= background_data

        return corrected_data


class FabioImageReader(Reader):
    """Reader for images using the python package.
    [`fabio`](https://fabio.readthedocs.io/en/main/).

    :ivar frame: Index of a specific frame to read from the file(s),
        defaults to `None`.
    :type frame: int, optional
    """
    frame: Optional[conint(ge=0)] = None

    def read(self):
        """Return the data from the image file(s) provided.

        :returns: Image data as a numpy array (or list of numpy
            arrays, if a glob pattern matching more than one file was
            provided).
        :rtype: Union[numpy.ndarray, list[numpy.ndarray]]
        """
        # Third party modules
        from glob import glob
        import fabio

        filenames = glob(self.filename)
        data = []
        for f in filenames:
            image = fabio.open(f, frame=self.frame)
            data.append(image.data)
            image.close()
        return data


class H5Reader(Reader):
    """Reader for h5 files.

    :ivar h5path: Path to a specific location in the h5 file to read
        data from, defaults to `'/'`.
    :type h5path: str, optional
    :ivar idx: Data slice to read from the object at the specified
        location in the h5 file.
    :type idx: list[int], optional

    """
    h5path: Optional[constr(strip_whitespace=True, min_length=1)] = '/'
    idx: Optional[conlist(min_length=1, max_length=3, item_type=int)] = None

    def read(self):
        """Return the data object stored at `h5path` in an h5-file.

        :return: Object indicated by `filename` and `h5path`.
        :rtype: object
        """
        # Third party modules
        from h5py import File

        data = File(self.filename, 'r')[self.h5path]
        if self.idx is not None:
            data = data[tuple(self.idx)]
        return data


class LinkamReader(Reader):
    """Reader for loading Linkam load frame .txt files as an
    `NXdata`.

    :ivar columns: Column names to read in, defaults to None
        (read in all columns)
    :type columns: list[str], optional
    """
    columns: Optional[conlist(
        item_type=constr(strip_whitespace=True, min_length=1))] = None

    def read(self):
        """Read specified columns from the given Linkam file.

        :returns: Linkam data represented in an `NXdata` object
        :rtype: nexusformat.nexus.NXdata
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        # Parse .txt file
        start_time, metadata, data = LinkamReader.parse_file(
            self.filename, self.logger)

        # Get list of actual data column names and corresponding
        # signal nxnames (same as user-supplied column names)
        signal_names = []
        if self.columns is None:
            signal_names = [(col, col) for col in data.keys() if col != 'Time']
        else:
            for col in self.columns:
                col_actual = col
                if col == 'Distance':
                    col_actual = 'Force V Distance_X'
                elif col == 'Force':
                    col_actual = 'Force V Distance_Y'
                elif not col in data:
                    if f'{col}_Y' in data:
                        # Always use the *_Y column if the user-supplied
                        # column name has both _X and _Y components
                        col_actual = f'{col}_Y'
                    else:
                        self.logger.warning(
                            f'{col} not present in {self.filename}')
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
            ) for col_actual, col in signal_names},
            attrs=metadata
        )
        return nxdata

    @classmethod
    def parse_file(cls, filename, logger):
        """Return start time, metadata, and data stored in the
        provided Linkam .txt file.

        :returns:
        :rtype: tuple(float, dict[str, str], dict[str, list[float]])
        """
        # System modules
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
            dt = datetime.strptime(datetime_str, '%d-%m-%y_%H-%M-%S-%f')
            start_time = dt.timestamp()
        else:
            logger.warning(f'Datetime not found in {filename}')

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
                    values = line.replace(',', '').split('\t')
                    for val, col in zip(values, list(data.keys())):
                        try:
                            val = float(val)
                        except Exception as exc:
                            logger.warning(
                                f'Cannot convert {col} value to float: {val} '
                                f'({exc})')
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
                if re.match(r'^([\w\s\w]+)(\t\t[\w\s\w]+)*$', line):
                    # Match found for start of data section -- this
                    # line and the next are column labels.
                    data_cols = []
                    # Get base quantity column names
                    base_cols = line.split('\t\t')
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
                    # First column (after 0th) is actually Time
                    data_cols[1] = 'Time'
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
        nxentry.map_config = map_config.model_dump_json()
        nxentry.spec_scans = NXcollection()
        for scans in map_config.spec_scans:
            nxentry.spec_scans[scans.scanparsers[0].scan_name] = \
                NXfield(value=scans.scan_numbers,
                        attrs={'spec_file': str(scans.spec_file)})

        # Add sample metadata
        nxentry[map_config.sample.name] = NXsample(
            **map_config.sample.model_dump())

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


class PandasReader(Reader):
    """Reader for files that can be read in with
    (pandas)[https://pandas.pydata.org/docs/index.html]"""
    def read(self, filename, method='read_csv', comment='#', kwargs=None):
        """Return a `pandas.DataFrame` read from the given file.

        :param filename: Name of file to read from.
        :type filename: str
        :param method: Name of `pandas` method to use for reading from
            `filename`. Defaults to `'read_csv'`.
        :type method: str, optional
        :param comment: Character to identify comment lines in the
            input file, defaults to `'#'`.
        :type comment: str, optional
        :param kwargs: Additional keyword arguments to supply to the
            `pandas` reader.
        :param kwargs: dict[str, object], optional.
        :rtype: `pandas.DataFrame`
        """
        # Third party modules
        import pandas as pd

        reader = getattr(pd, method)
        if not callable(reader):
            raise ValueError(
                f'{method} is not a callable pandas reader method')

        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise TypeError(
                f'Invalid kwargs type ({type(kwargs)}, should be dict)')

        return reader(filename, comment=comment, **kwargs)


class NexusReader(Reader):
    """Reader for NeXus files.

    :ivar nxpath: Path to a specific location in the NeXus file tree
        to read from, defaults to `'/'`.
    :type nxpath: str, optional
    :ivar idx: Index of array to select, defaults to `None`
    :type idx: int, optional
    :ivar mode: File mode, defaults to 'r'.
    :type mode: Literal['r', 'rw', 'r+', 'w', 'a'], optional
    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :type nxmemory: int, optional
    """
    nxpath: Optional[constr(strip_whitespace=True, min_length=1)] = '/'
    idx: Optional[conint(ge=0)] = None
    mode: Literal['r', 'rw', 'r+', 'w', 'a'] = 'r'
    nxmemory: Optional[conint(gt=0)] = None

    def read(self):
        """Return the NeXus object stored at `nxpath` in a NeXus file.

        :raises nexusformat.nexus.NeXusError: If `filename` is not a
            NeXus file or `nxpath` is not in its tree.
        :return: NeXus object indicated by `filename` and `nxpath`.
        :rtype: nexusformat.nexus.NXobject
        """
        # Third party modules
        from nexusformat.nexus import (
            nxload,
            nxsetconfig,
        )

        if self.nxmemory is not None:
            nxsetconfig(memory=self.nxmemory)
        if self.idx is not None:
            return nxload(self.filename, mode=self.mode)[self.nxpath][self.idx]
        return nxload(self.filename, mode=self.mode)[self.nxpath]


class NXdataReader(Reader):
    """Reader for constructing an NXdata object from components."""
    def read(self, name, nxfield_params, signal_name, axes_names, attrs=None):
        """Return a basic NXdata object constructed from components.

        :param name: NXdata group name.
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
        :returns: A new NXdata object.
        :rtype: nexusformat.nexus.NXdata
        """
        # Third party modules
        from nexusformat.nexus import NXdata

        # Read in NXfields
        nxfields = [NXfieldReader().read(**params, inputdir=self.inputdir)
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
    def read(self, nxpath, nxname=None, update_attrs=None, slice_params=None):
        """Return a copy of the indicated NXfield from the file. Name
        and attributes of the returned copy may be modified with the
        `nxname` and `update_attrs` keyword arguments.

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
        :returns: A copy of the indicated NXfield (with name and
            attributes optionally modified).
        :rtype: nexusformat.nexus.NXfield
        """
        # Third party modules
        from nexusformat.nexus import (
            NXfield,
            nxload,
        )

        nxroot = nxload(self.filename)
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
    """Reader for CHESS SPEC scans.

    :ivar config: SPEC configuration to be passed directly to the
        constructor of `CHAP.common.models.map.SpecConfig`.
    :type config: dict, optional
    :ivar detectors: Detector configurations of the detectors to
        include raw data for in the returned NeXus NXroot object,
        defaults to None (only a valid input for EDD).
    :type detectors: Union[
        dict, common.models.map.DetectorConfig], optional
    :ivar filename: Name of file to read from.
    :type filename: str, optional
    """
    config: Optional[Union[dict, SpecConfig]] = None
    detector_config: Optional[DetectorConfig] = None
    filename: Optional[str] = None

    _mapping_filename: PrivateAttr(default=None)

    _validate_filename = model_validator(mode='after')(validate_model)

    @model_validator(mode='after')
    def validate_specreader_after(self):
        """Validate the `SpecReader` configuration.

        :return: The validated configuration.
        :rtype: PipelineItem
        """
        if self.filename is not None:
            if self.config is not None:
                raise ValueError('Specify either filename or config in '
                       'common.SpecReader, not both')
            self.config = YAMLReader(**self.model_dump()).read()
        self.config = self.get_config(
            config=self.config, schema='common.models.map.SpecConfig')
        if self.detector_config is None:
            if self.config.experiment_type != 'EDD':
                raise RuntimeError(
                    'Missing parameter detector_config for experiment type '
                    f'{self.config.experiment_type}')
        return self

    def read(self):
        """Take a SPEC configuration filename or dictionary and return
        the raw data as a NeXus NXentry object.

        :return: Data from the provided SPEC configuration.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        # pylint: disable=no-name-in-module
        from json import dumps
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXentry,
            NXfield,
            NXroot,
        )
        # pylint: enable=no-name-in-module

        # Local modules
        from CHAP.common.models.map import Detector

        # Create the NXroot object
        nxroot = NXroot()
        nxentry = NXentry(name=self.config.experiment_type)
        nxroot[nxentry.nxname] = nxentry

        # Set up NXentry and add misc. CHESS-specific metadata as well
        # as all spec_motors, scan_columns, and smb_pars, and the
        # detector info and raw detector data
        nxentry.config = self.config.model_dump_json()
        nxentry.attrs['station'] = self.config.station
        nxentry.spec_scans = NXcollection()
#        nxpaths = []
        if self.config.experiment_type == 'EDD':
            detector_data_format = None
        for scans in self.config.spec_scans:
            nxscans = NXcollection()
            nxentry.spec_scans[f'{scans.scanparsers[0].scan_name}'] = nxscans
            nxscans.attrs['spec_file'] = str(scans.spec_file)
            nxscans.attrs['scan_numbers'] = scans.scan_numbers
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                if self.config.experiment_type == 'EDD':
                    if detector_data_format is None:
                        detector_data_format = scanparser.detector_data_format
                    elif (scanparser.detector_data_format !=
                            detector_data_format):
                        raise NotImplementedError(
                            'Mixing `spec` and `h5` data formats')
                    if self.detector_config is None:
                        detectors_ids = None
                    elif detector_data_format == 'spec':
                        raise NotImplementedError(
                            'detector_data_format = "spec"')
                    else:
                        detectors_ids = [
                            int(d.get_id())
                            for d in self.detector_config.detectors]
                nxscans[scan_number] = NXcollection()
                try:
                    nxscans[scan_number].spec_motors = dumps(
                        {k:float(v) for k,v
                         in scanparser.spec_positioner_values.items()})
                except Exception:
                    pass
                try:
                    nxscans[scan_number].scan_columns = dumps(
                        {k:list(v) for k,v
                         in scanparser.spec_scan_data.items() if len(v)})
                except Exception:
                    pass
                try:
                    nxscans[scan_number].smb_pars = dumps(
                        {k:v for k,v in scanparser.pars.items()})
                except Exception:
                    pass
                try:
                    nxscans[scan_number].spec_scan_motor_mnes = dumps(
                        scanparser.spec_scan_motor_mnes)
                except Exception:
                    pass
                if self.config.experiment_type == 'EDD':
                    nxdata = NXdata()
                    nxscans[scan_number].data = nxdata
#                    nxpaths.append(
#                        f'spec_scans/{nxscans.nxname}/{scan_number}/data')
                    nxdata.data = NXfield(
                        value=scanparser.get_detector_data(detectors_ids)[0])
                else:
                    if self.config.experiment_type == 'TOMO':
                        dtype = np.float32
                    else:
                        dtype = None
                    nxdata = NXdata()
                    nxscans[scan_number].data = nxdata
#                    nxpaths.append(
#                        f'spec_scans/{nxscans.nxname}/{scan_number}/data')
                    for detector in self.detector_config.detectors:
                        nxdata[detector.get_id()] = NXfield(
                           value=scanparser.get_detector_data(
                               detector.get_id(), dtype=dtype))

        if (self.config.experiment_type == 'EDD' and
                self.detector_config is None):
            if detector_data_format == 'spec':
                raise NotImplementedError('detector_data_format = "spec"')
            self.detector_config = DetectorConfig(
                detectors=[
                    Detector(id=i) for i in range(nxdata.data.shape[1])])
        nxentry.detectors = self.detector_config.model_dump_json()

        #return nxroot, nxpaths
        return nxroot


class URLReader(Reader):
    """Reader for data available over HTTPS."""
    def read(self, url, headers=None, timeout=10):
        """Make an HTTPS request to the provided URL and return the
        results. Headers for the request are optional.

        :param url: URL to read.
        :type url: str
        :param headers: Headers to attach to the request.
        :type headers: dict, optional
        :param timeout: Timeout for the HTTPS request,
            defaults to `10`.
        :type timeout: int
        :return: Content of the response.
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
    def read(self):
        """Return a dictionary from the contents of a yaml file.

        :return: Contents of the file.
        :rtype: dict
        """
        # Third party modules
        import yaml

        with open(self.filename) as f:
            data = yaml.safe_load(f)
        return data

class ZarrReader(Reader):
    """Reader for Zarr stores."""

    def read(self, filename, path='/', idx=None, mode='r'):
        """Return the Zarr object stored at `path` in a Zarr store.

        :param filename: Path or URL to the Zarr store.
        :type filename: str
        :param path: Path to a specific location in the Zarr hierarchy
            to read from, defaults to `'/'`.
        :type path: str, optional
        :param idx: Optional index or slice to apply to the returned object.
        :type idx: int, slice, tuple, optional
        :param mode: Store access mode, defaults to `'r'`.
            Common values: `'r'`, `'r+'`, `'a'`, `'w'`.
        :type mode: str, optional
        :raises KeyError: If `path` does not exist in the store.
        :return: The Zarr array or group indicated by `filename` and `path`.
        :rtype: zarr.core.Array or zarr.hierarchy.Group
        """
        import zarr

        # Open the Zarr store (directory, zip, or URL)
        root = zarr.open(filename, mode=mode)

        # Normalize path handling
        if path == '/' or path == '':
            data = root
        else:
            # Remove leading slash for Zarr traversal
            data = root[path.lstrip('/')]

        # Optional indexing (arrays only)
        if idx is not None:
            data = data[idx]

        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.reader import main

    main()
