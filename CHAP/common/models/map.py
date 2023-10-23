# System modules
from copy import deepcopy
from functools import (
    cache,
    lru_cache,
)
import os
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
import numpy as np
from pydantic import (
    BaseModel,
    conint,
    conlist,
    constr,
    FilePath,
    PrivateAttr,
    root_validator,
    validator,
)
from pyspec.file.spec import FileSpec


class Sample(BaseModel):
    """Class representing a sample metadata configuration.

    :ivar name: The name of the sample.
    :type name: str
    :ivar description: A description of the sample.
    :type description: Optional[str]
    """
    name: constr(min_length=1)
    description: Optional[str]


class SpecScans(BaseModel):
    """Class representing a set of scans from a single SPEC file.

    :ivar spec_file: Path to the SPEC file.
    :type spec_file: str
    :ivar scan_numbers: List of scan numbers to use.
    :type scan_numbers: list[int]
    """
    spec_file: FilePath
    scan_numbers: conlist(item_type=conint(gt=0), min_items=1)

    @validator('spec_file', allow_reuse=True)
    def validate_spec_file(cls, spec_file, values):
        """Validate the specified SPEC file.

        :param spec_file: Path to the SPEC file.
        :type spec_file: str
        :raises ValueError: If the SPEC file is invalid.
        :return: Absolute path to the SPEC file, if it is valid.
        :rtype: str
        """
        try:
            spec_file = os.path.abspath(spec_file)
            sspec_file = FileSpec(spec_file)
        except:
            raise ValueError(f'Invalid SPEC file {spec_file}')
        return spec_file

    @validator('scan_numbers', pre=True, allow_reuse=True)
    def validate_scan_numbers(cls, scan_numbers, values):
        """Validate the specified list of scan numbers.

        :param scan_numbers: List of scan numbers.
        :type scan_numbers: list of int
        :param values: Dictionary of values for all fields of the model.
        :type values: dict
        :raises ValueError: If a specified scan number is not found in
            the SPEC file.
        :return: List of scan numbers.
        :rtype: list of int
        """
        if isinstance(scan_numbers, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_numbers = string_to_list(scan_numbers)

        spec_file = values.get('spec_file')
        if spec_file is not None:
            spec_scans = FileSpec(spec_file)
            for scan_number in scan_numbers:
                scan = spec_scans.get_scan_by_number(scan_number)
                if scan is None:
                    raise ValueError(
                        f'No scan number {scan_number} in {spec_file}')
        return scan_numbers

    @property
    def scanparsers(self):
        """A list of `ScanParser`s for each of the scans specified by
        the SPEC file and scan numbers belonging to this instance of
        `SpecScans`
        """
        return [self.get_scanparser(scan_no) for scan_no in self.scan_numbers]

    def get_scanparser(self, scan_number):
        """This method returns a `ScanParser` for the specified scan
        number in the specified SPEC file.

        :param scan_number: Scan number to get a `ScanParser` for
        :type scan_number: int
        :return: `ScanParser` for the specified scan number
        :rtype: ScanParser
        """
        return get_scanparser(self.spec_file, scan_number)

    def get_index(self, scan_number:int, scan_step_index:int, map_config):
        """This method returns a tuple representing the index of a
        specific step in a specific SPEC scan within a map.

        :param scan_number: Scan number to get index for
        :type scan_number: int
        :param scan_step_index: Scan step index to get index for
        :type scan_step_index: int
        :param map_config: Map configuration to get index for
        :type map_config: MapConfig
        :return: Index for the specified scan number and scan step
            index within the specified map configuration
        :rtype: tuple
        """
        index = ()
        for independent_dimension in map_config.independent_dimensions:
            coordinate_index = list(
                map_config.coords[independent_dimension.label]).index(
                    independent_dimension.get_value(
                        self, scan_number, scan_step_index))
            index = (coordinate_index, *index)
        return index

    def get_detector_data(self,
                          detectors:list,
                          scan_number:int,
                          scan_step_index:int):
        """Return the raw data from the specified detectors at the
        specified scan number and scan step index.

        :param detectors: List of detector prefixes to get raw data for
        :type detectors: list[str]
        :param scan_number: Scan number to get data for
        :type scan_number: int
        :param scan_step_index: Scan step index to get data for
        :type scan_step_index: int
        :return: Data from the specified detectors for the specified
            scan number and scan step index
        :rtype: list[np.ndarray]
        """
        return get_detector_data(
            tuple([detector.prefix for detector in detectors]),
            self.spec_file,
            scan_number,
            scan_step_index)


@cache
def get_available_scan_numbers(spec_file:str):
    scans = FileSpec(spec_file).scans
    scan_numbers = list(scans.keys())
    return scan_numbers


@cache
def get_scanparser(spec_file:str, scan_number:int):
    if scan_number not in get_available_scan_numbers(spec_file):
        return None
    return ScanParser(spec_file, scan_number)


@lru_cache(maxsize=10)
def get_detector_data(
        detector_prefixes:tuple,
        spec_file:str,
        scan_number:int,
        scan_step_index:int):
    detector_data = []
    scanparser = get_scanparser(spec_file, scan_number)
    for prefix in detector_prefixes:
        image_data = scanparser.get_detector_data(prefix, scan_step_index)
        detector_data.append(image_data)
    return detector_data


class PointByPointScanData(BaseModel):
    """Class representing a source of raw scalar-valued data for which
    a value was recorded at every point in a `MapConfig`.

    :ivar label: A user-defined label for referring to this data in
        the NeXus file and in other tools.
    :type label: str
    :ivar units: The units in which the data were recorded.
    :type units: str
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['spec_motor', 'scan_column', 'smb_par']
    :ivar name: Represents the name with which these raw data were
        recorded at time of data collection.
    :type name: str
    """
    label: constr(min_length=1)
    units: constr(strip_whitespace=True, min_length=1)
    data_type: Literal['spec_motor', 'scan_column', 'smb_par']
    name: constr(strip_whitespace=True, min_length=1)

    @validator('label')
    def validate_label(cls, label):
        """Validate that the supplied `label` does not conflict with
        any of the values for `label` reserved for certain data needed
        to perform corrections.

        :param label: The value of `label` to validate
        :type label: str
        :raises ValueError: If `label` is one of the reserved values.
        :return: The original supplied value `label`, if it is
            allowed.
        :rtype: str
        """
        if ((not issubclass(cls,CorrectionsData))
                and label in CorrectionsData.reserved_labels()):
            raise ValueError(
                f'{cls.__name__}.label may not be any of the following '
                f'reserved values: {CorrectionsData.reserved_labels()}')
        return label

    def validate_for_station(self, station:str):
        """Validate this instance of `PointByPointScanData` for a
        certain choice of station (beamline).

        :param station: The name of the station (in 'idxx' format).
        :type station: str
        :raises TypeError: If the station is not compatible with the
            value of the `data_type` attribute for this instance of
            PointByPointScanData.
        :return: None
        :rtype: None
        """
        if (station.lower() not in ('id1a3', 'id3a')
                and self.data_type == 'smb_par'):
            raise TypeError(
                f'{self.__class__.__name__}.data_type may not be "smb_par" '
                f'when station is "{station}"')

    def validate_for_spec_scans(
            self, spec_scans:list[SpecScans],
            scan_step_index:Union[Literal['all'],int] = 'all'):
        """Validate this instance of `PointByPointScanData` for a list
        of `SpecScans`.

        :param spec_scans: A list of `SpecScans` whose raw data will
            be checked for the presence of the data represented by
            this instance of `PointByPointScanData`
        :type spec_scans: list[SpecScans]
        :param scan_step_index: A specific scan step index to validate,
            defaults to `'all'`.
        :type scan_step_index: Union[Literal['all'],int], optional
        :raises RuntimeError: If the data represented by this instance of
            `PointByPointScanData` is missing for the specified scan steps.
        :return: None
        :rtype: None
        """
        for scans in spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                if scan_step_index == 'all':
                    scan_step_index_range = range(scanparser.spec_scan_npts)
                else:
                    scan_step_index_range = range(
                        scan_step_index, 1+scan_step_index)
                for index in scan_step_index_range:
                    try:
                        self.get_value(scans, scan_number, index)
                    except:
                        raise RuntimeError(
                            f'Could not find data for {self.name} '
                            f'(data_type "{self.data_type}") '
                            f'on scan number {scan_number} '
                            f'for index {index} '
                            f'in spec file {scans.spec_file}')

    def get_value(self, spec_scans:SpecScans,
                  scan_number:int, scan_step_index:int=0):
        """Return the value recorded for this instance of
        `PointByPointScanData` at a specific scan step.

        :param spec_scans: An instance of `SpecScans` in which the
            requested scan step occurs.
        :type spec_scans: SpecScans
        :param scan_number: The number of the scan in which the
            requested scan step occurs.
        :type scan_number: int
        :param scan_step_index: The index of the requested scan step.
        :type scan_step_index: int
        :return: The value recorded of the data represented by this
            instance of `PointByPointScanData` at the scan step
            requested
        :rtype: float
        """
        if self.data_type == 'spec_motor':
            return get_spec_motor_value(spec_scans.spec_file,
                                        scan_number,
                                        scan_step_index,
                                        self.name)
        if self.data_type == 'scan_column':
            return get_spec_counter_value(spec_scans.spec_file,
                                          scan_number,
                                          scan_step_index,
                                          self.name)
        if self.data_type == 'smb_par':
            return get_smb_par_value(spec_scans.spec_file,
                                     scan_number,
                                     self.name)
        return None


@cache
def get_spec_motor_value(spec_file:str, scan_number:int,
                         scan_step_index:int, spec_mnemonic:str):
    """Return the value recorded for a SPEC motor at a specific scan
    step.

    :param spec_file: Location of a SPEC file in which the requested
        scan step occurs.
    :type spec_scans: str
    :param scan_number: The number of the scan in which the requested
        scan step occurs.
    :type scan_number: int
    :param scan_step_index: The index of the requested scan step.
    :type scan_step_index: int
    :param spec_mnemonic: The menmonic of a SPEC motor.
    :type spec_mnemonic: str
    :return: The value of the motor at the scan step requested
    :rtype: float
    """
    scanparser = get_scanparser(spec_file, scan_number)
    if (hasattr(scanparser, 'spec_scan_motor_mnes')
            and spec_mnemonic in scanparser.spec_scan_motor_mnes):
        motor_i = scanparser.spec_scan_motor_mnes.index(spec_mnemonic)
        if scan_step_index >= 0:
            scan_step = np.unravel_index(
                scan_step_index,
                scanparser.spec_scan_shape,
                order='F')
            motor_value = \
                scanparser.spec_scan_motor_vals[motor_i][scan_step[motor_i]]
        else:
            motor_value = scanparser.spec_scan_motor_vals[motor_i]
    else:
        motor_value = scanparser.get_spec_positioner_value(spec_mnemonic)
    return motor_value


@cache
def get_spec_counter_value(spec_file:str, scan_number:int,
                           scan_step_index:int, spec_column_label:str):
    """Return the value recorded for a SPEC counter at a specific scan
    step.

    :param spec_file: Location of a SPEC file in which the requested
        scan step occurs.
    :type spec_scans: str
    :param scan_number: The number of the scan in which the requested
        scan step occurs.
    :type scan_number: int
    :param scan_step_index: The index of the requested scan step.
    :type scan_step_index: int
    :param spec_column_label: The label of a SPEC data column.
    :type spec_column_label: str
    :return: The value of the counter at the scan step requested
    :rtype: float
    """
    scanparser = get_scanparser(spec_file, scan_number)
    if scan_step_index >= 0:
        return scanparser.spec_scan_data[spec_column_label][scan_step_index]
    return scanparser.spec_scan_data[spec_column_label]


@cache
def get_smb_par_value(spec_file:str, scan_number:int, par_name:str):
    """Return the value recorded for a specific scan in SMB-tyle .par
    file.

    :param spec_file: Location of a SPEC file in which the requested
        scan step occurs.
    :type spec_scans: str
    :param scan_number: The number of the scan in which the requested
        scan step occurs.
    :type scan_number: int
    :param par_name: The name of the column in the .par file
    :type par_name: str
    :return: The value of the .par file value for  the scan requested.
    :rtype: float
    """
    scanparser = get_scanparser(spec_file, scan_number)
    return scanparser.pars[par_name]


def validate_data_source_for_map_config(data_source, values):
    """Confirm that an instance of PointByPointScanData is valid for
    the station and scans provided by a map configuration dictionary.

    :param data_source: the input object to validate
    :type data_source: PintByPointScanData
    :param values: the map configuration dictionary
    :type values: dict
    :raises Exception: if `data_source` cannot be validated for
        `values`.
    :return: `data_source`, iff it is valid.
    :rtype: PointByPointScanData
    """
    if data_source is not None:
        import_scanparser(values.get('station'), values.get('experiment_type'))
        data_source.validate_for_station(values.get('station'))
        data_source.validate_for_spec_scans(values.get('spec_scans'))
    return data_source


class IndependentDimension(PointByPointScanData):
    """Class representing the source of data to identify the
    coordinate values along one dimension of a `MapConfig`

    :ivar label: A user-defined label for referring to this data in
        the NeXus file and in other tools.
    :type label: str
    :ivar units: The units in which the data were recorded.
    :type units: str
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['spec_motor', 'scan_column', 'smb_par']
    :ivar name: Represents the name with which these raw data were
        recorded at time of data collection.
    :type name: str
    :param start: Sarting index for slicing all datasets of a
        `MapConfig` along this axis, defaults to 0
    :type start: int, optional
    :param end: Ending index for slicing all datasets of a `MapConfig`
        along this axis, defaults to the total number of unique values
        along this axis in the associated `MapConfig`
    :type end: int, optional
    :param step: Step for slicing all datasets of a `MapConfig` along
        this axis, defaults to 1
    :type step: int, optional
    """
    start: Optional[conint(ge=0)] = 0
    end: Optional[int] = None
    step: Optional[conint(gt=0)] = 1

#    @validator('step')
#    def validate_step(cls, value):
#        if value == 0 :
#            raise ValueError('slice step cannot be zero')
#        return value


class CorrectionsData(PointByPointScanData):
    """Class representing the special instances of
    `PointByPointScanData` that are used by certain kinds of
    `CorrectionConfig` tools.

    :ivar label: One of the reserved values required by
        `CorrectionConfig`, `'presample_intensity'`,
        `'postsample_intensity'`, or `'dwell_time_actual'`.
    :type label: Literal['presample_intensity',
                         'postsample_intensity',
                         'dwell_time_actual']
    :ivar units: The units in which the data were recorded.
    :type units: str
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['scan_column', 'smb_par']
    :ivar name: Represents the name with which these raw data were
        recorded at time of data collection.
    :type name: str
    """
    label: Literal['presample_intensity',
                   'postsample_intensity',
                   'dwell_time_actual']
    data_type: Literal['scan_column','smb_par']

    @classmethod
    def reserved_labels(cls):
        """Return a list of all the labels reserved for
        corrections-related scalar data.

        :return: A list of reserved labels
        :rtype: list[str]
        """
        return list(cls.__fields__['label'].type_.__args__)


class PresampleIntensity(CorrectionsData):
    """Class representing a source of raw data for the intensity of
    the beam that is incident on the sample.

    :ivar label: Must be `"presample_intensity"`
    :type label: Literal["presample_intensity"]
    :ivar units: Must be `"counts"`
    :type units: Literal["counts"]
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['scan_column', 'smb_par']
    :ivar name: Represents the name with which these raw data were
        recorded at time of data collection.
    :type name: str
    """
    label: Literal['presample_intensity'] = 'presample_intensity'
    units: Literal['counts'] = 'counts'


class PostsampleIntensity(CorrectionsData):
    """Class representing a source of raw data for the intensity of
    the beam that has passed through the sample.

    :ivar label: Must be `"postsample_intensity"`
    :type label: Literal["postsample_intensity"]
    :ivar units: Must be `"counts"`
    :type units: Literal["counts"]
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['scan_column', 'smb_par']
    :ivar name: Represents the name with which these raw data were
        recorded at time of data collection.
    :type name: str
    """
    label: Literal['postsample_intensity'] = 'postsample_intensity'
    units: Literal['counts'] = 'counts'


class DwellTimeActual(CorrectionsData):
    """Class representing a source of raw data for the actual dwell
    time at each scan point in SPEC (with some scan types, this value
    can vary slightly point-to-point from the dwell time specified in
    the command).

    :ivar label: Must be `"dwell_time_actual"`
    :type label: Literal["dwell_time_actual"]
    :ivar units: Must be `"counts"`
    :type units: Literal["counts"]
    :ivar data_type: Represents how these data were recorded at time
        of data collection.
    :type data_type: Literal['scan_column', 'smb_par']
    :ivar name: Represents the name with which these raw data were
        recorded at time of data collection.
    :type name: str
    """
    label: Literal['dwell_time_actual'] = 'dwell_time_actual'
    units: Literal['s'] = 's'


class SpecConfig(BaseModel):
    """Class representing the raw data for one or more SPEC scans.

    :ivar station: The name of the station at which the data was
        collected.
    :type station: Literal['id1a3','id3a','id3b']
    :ivar spec_scans: A list of the SPEC scans that compose the set.
    :type spec_scans: list[SpecScans]
    """
    station: Literal['id1a3','id3a','id3b']
    experiment_type: Literal['SAXSWAXS', 'EDD', 'XRF', 'TOMO']
    spec_scans: conlist(item_type=SpecScans, min_items=1)

    @root_validator(pre=True)
    def validate_config(cls, values):
        """Ensure that a valid configuration was provided and finalize
        spec_file filepaths.

        :param values: Dictionary of class field values.
        :type values: dict
        :return: The validated list of `values`.
        :rtype: dict
        """
        # System modules
        from copy import deepcopy

        inputdir = values.get('inputdir')
        if inputdir is not None:
            spec_scans = values.get('spec_scans')
            for i, scans in enumerate(deepcopy(spec_scans)):
                spec_file = scans['spec_file']
                if not os.path.isabs(spec_file):
                    spec_scans[i]['spec_file'] = os.path.join(
                        inputdir, spec_file)
                spec_scans[i] = SpecScans(**spec_scans[i], **values)
            values['spec_scans'] = spec_scans
        return values

    @validator('experiment_type')
    def validate_experiment_type(cls, value, values):
        """Ensure values for the station and experiment_type fields are
        compatible
        """
        station = values.get('station')
        if station == 'id1a3':
            allowed_experiment_types = ['SAXSWAXS', 'EDD', 'TOMO']
        elif station == 'id3a':
            allowed_experiment_types = ['EDD', 'TOMO']
        elif station == 'id3b':
            allowed_experiment_types = ['SAXSWAXS', 'XRF', 'TOMO']
        else:
            allowed_experiment_types = []
        if value not in allowed_experiment_types:
            raise ValueError(
                f'For station {station}, allowed experiment types are '
                f'{", ".join(allowed_experiment_types)}. '
                f'Supplied experiment type {value} is not allowed.')
        import_scanparser(station, value)
        return value


class MapConfig(BaseModel):
    """Class representing an experiment consisting of one or more SPEC
    scans.

    :ivar title: The title for the map configuration.
    :type title: str
    :ivar station: The name of the station at which the map was
        collected.
    :type station: Literal['id1a3','id3a','id3b']
    :ivar spec_scans: A list of the SPEC scans that compose the map.
    :type spec_scans: list[SpecScans]
    :ivar independent_dimensions: A list of the sources of data
        representing the raw values of each independent dimension of
        the map.
    :type independent_dimensions: list[PointByPointScanData]
    :ivar presample_intensity: A source of point-by-point presample
        beam intensity data. Required when applying a CorrectionConfig
        tool.
    :type presample_intensity: Optional[PresampleIntensity]
    :ivar dwell_time_actual: A source of point-by-point actual dwell
        times for SPEC scans. Required when applying a
        CorrectionConfig tool.
    :type dwell_time_actual: Optional[DwellTimeActual]
    :ivar presample_intensity: A source of point-by-point postsample
        beam intensity data. Required when applying a CorrectionConfig
        tool with `correction_type="flux_absorption"` or
        `correction_type="flux_absorption_background"`.
    :type presample_intensity: Optional[PresampleIntensity]
    :ivar scalar_data: A list of the sources of data representing
        other scalar raw data values collected at each point on the
        map. In the NeXus file representation of the map, datasets for
        these values will be included.
    :type scalar_values: Optional[list[PointByPointScanData]]
    :ivar map_type: Type of map, structured or unstructured,
        defaults to `'structured'`.
    :type map_type: Optional[Literal['structured', 'unstructured']]
    """
    title: constr(strip_whitespace=True, min_length=1)
    station: Literal['id1a3','id3a','id3b']
    experiment_type: Literal['SAXSWAXS', 'EDD', 'XRF', 'TOMO']
    sample: Sample
    spec_scans: conlist(item_type=SpecScans, min_items=1)
    independent_dimensions: conlist(
        item_type=IndependentDimension, min_items=1)
    presample_intensity: Optional[PresampleIntensity]
    dwell_time_actual: Optional[DwellTimeActual]
    postsample_intensity: Optional[PostsampleIntensity]
    scalar_data: Optional[list[PointByPointScanData]] = []
    map_type: Optional[Literal['structured', 'unstructured']] = 'structured'
    _coords: dict = PrivateAttr()
    _dims: tuple = PrivateAttr()
    _scan_step_indices: list = PrivateAttr()
    _shape: tuple = PrivateAttr()

    _validate_independent_dimensions = validator(
        'independent_dimensions',
        each_item=True,
        allow_reuse=True)(validate_data_source_for_map_config)
    _validate_presample_intensity = validator(
        'presample_intensity',
        allow_reuse=True)(validate_data_source_for_map_config)
    _validate_dwell_time_actual = validator(
        'dwell_time_actual',
        allow_reuse=True)(validate_data_source_for_map_config)
    _validate_postsample_intensity = validator(
        'postsample_intensity',
        allow_reuse=True)(validate_data_source_for_map_config)
    _validate_scalar_data = validator(
        'scalar_data',
        each_item=True,
        allow_reuse=True)(validate_data_source_for_map_config)

    @root_validator(pre=True)
    def validate_config(cls, values):
        """Ensure that a valid configuration was provided and finalize
        spec_file filepaths.

        :param values: Dictionary of class field values.
        :type values: dict
        :return: The validated list of `values`.
        :rtype: dict
        """
        inputdir = values.get('inputdir')
        if inputdir is not None:
            spec_scans = values.get('spec_scans')
            for i, scans in enumerate(deepcopy(spec_scans)):
                spec_file = scans['spec_file']
                if not os.path.isabs(spec_file):
                    spec_scans[i]['spec_file'] = os.path.join(
                        inputdir, spec_file)
                spec_scans[i] = SpecScans(**spec_scans[i], **values)
            values['spec_scans'] = spec_scans
        return values

    @validator('map_type', pre=True, always=True)
    def validate_map_type(cls, map_type, values):
        """Validate the map_type field.

        :param map_type: Type of map, structured or unstructured,
            defaults to `'structured'`.
        :type map_type: Literal['structured', 'unstructured']]
        :param values: Dictionary of values for all fields of the model.
        :type values: dict
        :return: The validated value for map_type.
        :rtype: str
        """
        dims = {}
        spec_scans = values.get('spec_scans')
        independent_dimensions = values.get('independent_dimensions')
        import_scanparser(values.get('station'), values.get('experiment_type'))
        for i, dim in enumerate(deepcopy(independent_dimensions)):
            dims[dim.label] = []
            for scans in spec_scans:
                for scan_number in scans.scan_numbers:
                    scanparser = scans.get_scanparser(scan_number)
                    for scan_step_index in range(
                            scanparser.spec_scan_npts):
                        dims[dim.label].append(dim.get_value(
                                scans, scan_number, scan_step_index))
            dims[dim.label] = np.unique(dims[dim.label])
            if dim.end is None:
                dim.end = len(dims[dim.label])
            dims[dim.label] = dims[dim.label][slice(
                dim.start, dim.end, dim.step)]
            independent_dimensions[i] = dim

        coords = np.zeros([v.size for v in dims.values()], dtype=np.int64)
        for scans in spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    coords[tuple([
                        list(dims[dim.label]).index(
                            dim.get_value(scans, scan_number, scan_step_index))
                        for dim in independent_dimensions])] += 1
        if any(True for v in coords.flatten() if v == 0 or v > 1):
            return 'unstructured'
        else:
            return 'structured'


    @validator('experiment_type')
    def validate_experiment_type(cls, value, values):
        """Ensure values for the station and experiment_type fields are
        compatible
        """
        station = values.get('station')
        if station == 'id1a3':
            allowed_experiment_types = ['SAXSWAXS', 'EDD', 'TOMO']
        elif station == 'id3a':
            allowed_experiment_types = ['EDD', 'TOMO']
        elif station == 'id3b':
            allowed_experiment_types = ['SAXSWAXS', 'XRF', 'TOMO']
        else:
            allowed_experiment_types = []
        if value not in allowed_experiment_types:
            raise ValueError(
                f'For station {station}, allowed experiment types are '
                f'{", ".join(allowed_experiment_types)}. '
                f'Supplied experiment type {value} is not allowed.')
        return value

    @property
    def all_scalar_data(self):
        """Return a list of all instances of `PointByPointScanData`
        for which this map configuration will collect dataset-like
        data (as opposed to axes-like data).

        This will be any and all of the items in the
        corrections-data-related fields, as well as any additional
        items in the optional `scalar_data` field.
        """
        return [getattr(self, label, None)
                for label in CorrectionsData.reserved_labels()
                if getattr(self, label, None) is not None] + self.scalar_data

    @property
    def coords(self):
        """Return a dictionary of the values of each independent
        dimension across the map.
        """
        if not hasattr(self, "_coords"):
            coords = {}
            for dim in self.independent_dimensions:
                coords[dim.label] = []
                for scans in self.spec_scans:
                    for scan_number in scans.scan_numbers:
                        scanparser = scans.get_scanparser(scan_number)
                        for scan_step_index in range(
                                scanparser.spec_scan_npts):
                            coords[dim.label].append(dim.get_value(
                                    scans, scan_number, scan_step_index))
                if self.map_type == 'structured':
                    coords[dim.label] = np.unique(coords[dim.label])
            self._coords = coords
        return self._coords

    @property
    def dims(self):
        """Return a tuple of the independent dimension labels for the
        map.
        """
        if not hasattr(self, "_dims"):
            self._dims = [
                dim.label for dim in self.independent_dimensions[::-1]]
        return self._dims

    @property
    def scan_step_indices(self):
        """Return an ordered list in which we can look up the SpecScans
        object, the scan number, and scan step index for every point
        on the map.
        """
        if not hasattr(self, "_scan_step_indices"):
            scan_step_indices = []
            for scans in self.spec_scans:
                for scan_number in scans.scan_numbers:
                    scanparser = scans.get_scanparser(scan_number)
                    for scan_step_index in range(scanparser.spec_scan_npts):
                        scan_step_indices.append(
                            (scans, scan_number, scan_step_index))
            self._scan_step_indices = scan_step_indices
        return self._scan_step_indices

    @property
    def shape(self):
        """Return the shape of the map -- a tuple representing the
        number of unique values of each dimension across the map.
        """
        if not hasattr(self, "_shape"):
            if self.map_type == 'structured':
                self._shape = tuple(
                    [len(v) for k, v in self.coords.items()][::-1])
            else:
                self._shape =  (len(self.scan_step_indices),) 
        return self._shape

    def get_coords(self, map_index):
        """Return a dictionary of the coordinate names and values of
        each independent dimension for a given point on the map.

        :param map_index: The map index to return coordinates for.
        :type map_index: tuple
        :return: A list of coordinate values.
        :rtype: dict
        """
        if self.map_type == 'structured':
            return {dim:self.coords[dim][i]
                    for dim, i in zip(self.dims, map_index)}
        else:
            return {dim:self.coords[dim][map_index[0]] for dim in self.dims}

    def get_detector_data(self, detector_name, map_index):
        """Return detector data collected by this map for a given
        point on the map.

        :param detector_name: Name of the detector for which to return
            data. Usually the value of the detector's EPICS
            areaDetector prefix macro, $P.
        :type detector_name: str
        :param map_index: The map index to return detector data for.
        :type map_index: tuple
        :return: One frame of raw detector data.
        :rtype: np.ndarray
        """
        scans, scan_number, scan_step_index = \
            self.get_scan_step_index(map_index)
        scanparser = scans.get_scanparser(scan_number)
        return scanparser.get_detector_data(detector_name, scan_step_index)

    def get_scan_step_index(self, map_index):
        """Return parameters to identify a single SPEC scan step that
        corresponds to the map point at the index provided.

        :param map_index: The index of a map point to identify as a
            specific SPEC scan step index.
        :type map_index: tuple
        :return: A `SpecScans` configuration, scan number, and scan
            step index.
        :rtype: tuple[SpecScans, int, int]
        """
        if self.map_type == 'structured':
            map_coords = self.get_coords(map_index)
            for scans, scan_number, scan_step_index in self.scan_step_indices:
                coords = {dim.label:dim.get_value(
                               scans, scan_number, scan_step_index)
                           for dim in self.independent_dimensions}
                if coords == map_coords:
                    return scans, scan_number, scan_step_index
            raise RuntimeError(f'Unable to match coordinates {coords}')
        else:
            return self.scan_step_indices[map_index[0]]

    def get_value(self, data, map_index):
        """Return the raw data collected by a single device at a
        single point in the map.

        :param data: The device configuration to return a value of raw
            data for
        :type data: PointByPointScanData
        :param map_index: The map index to return raw data for
        :type map_index: tuple
        :return: Raw data value
        """
        scans, scan_number, scan_step_index = \
            self.get_scan_step_index(map_index)
        return data.get_value(scans, scan_number, scan_step_index)


def import_scanparser(station, experiment):
    """Given the name of a CHESS station and experiment type, import
    the corresponding subclass of `ScanParser` as `ScanParser`.

    :param station: The station name ("IDxx", not the beamline acronym)
    :type station: str
    :param experiment: The experiment type
    :type experiment: Literal["SAXSWAXS","EDD","XRF","Tomo","Powder"]
    :return: None
    """

    station = station.lower()
    experiment = experiment.lower()

    # Local modules
    if station in ('id1a3', 'id3a'):
        if experiment in ('saxswaxs', 'powder'):
            from CHAP.utils.scanparsers \
                import SMBLinearScanParser as ScanParser
        elif experiment == 'edd':
            from CHAP.utils.scanparsers \
                import SMBMCAScanParser as ScanParser
        elif experiment == 'tomo':
            from CHAP.utils.scanparsers \
                import SMBRotationScanParser as ScanParser
        else:
            raise ValueError(
                f'Invalid experiment type for station {station}: {experiment}')
    elif station == 'id3b':
        if experiment == 'saxswaxs':
            from CHAP.utils.scanparsers \
                import FMBSAXSWAXSScanParser as ScanParser
        elif experiment == 'tomo':
            from CHAP.utils.scanparsers \
                import FMBRotationScanParser as ScanParser
        elif experiment == 'xrf':
            from CHAP.utils.scanparsers \
                import FMBXRFScanParser as ScanParser
        else:
            raise ValueError(
                f'Invalid experiment type for station {station}: {experiment}')
    else:
        raise ValueError(f'Invalid station: {station}')

    globals()['ScanParser'] = ScanParser
