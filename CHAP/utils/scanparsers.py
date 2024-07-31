#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Parsing data from certain CHESS SPEC scans is supported by a family
of classes derived from the base class, `ScanParser` (defined
below). An instance of `ScanParser` represents a single SPEC scan --
each instance is initialized with the name of a specific spec file and
integer scan number. Access to certain data collected by that scan
(counter data, positioner values, scan shape, detector data, etc.) are
made available through the properties and methods of that object.

`ScanParser` is just an incomplete abstraction -- one should not
declare or work with an instance of `ScanParser` directly. Instead,
one must find the appropriate concrete subclass to use for the
particular type of scan one wishes to parse, then declare an instance
of that specific class to begin accessing data from that scan.

Basic usage examples:
1. Print out the position of a the SPEC positioner motor with mnemonic
`'mne0'` for a SAXS/WAXS scan collected at FMB:
    ```python
    from CHAP.utils.scanparsers import FMBSAXSWAXSScanParser
    sp = FMBSAXSWAXSScanParser('/path/to/fmb/saxswaxs/spec/file', 1)
    print(sp.get_spec_positioner_value('mne0'))
    ```
1. Store all the detector data collected by the detector with prefix
`'det'` over a rotation series collected at SMB in the variable
`data`:
    ```python
    from CHAP.utils.scanparsers import SMBRotationScanParser
    sp = SMBRotationScanParser('/path/to/smb/rotation/spec/file', 1)
    data = sp.get_detector_data('det')
    ```
"""

# System modules
from csv import reader
from fnmatch import filter as fnmatch_filter
from functools import cache
from json import load
import os
import re

# Third party modules
import numpy as np
from pyspec.file.spec import FileSpec
from pyspec.file.tiff import TiffFile

@cache
def filespec(spec_file_name):
    return FileSpec(spec_file_name)

class ScanParser:
    """Partial implementation of a class representing a SPEC scan and
    some of its metadata.

    :param spec_file_name: path to a SPEC file on the CLASSE DAQ
    :type spec_file_name: str
    :param scan_number: the number of a scan in the SPEC file provided
        with `spec_file_name`
    :type scan_number: int
    """

    def __init__(self,
                 spec_file_name:str,
                 scan_number:int):
        """Constructor method"""

        self.spec_file_name = spec_file_name
        self.scan_number = scan_number

        self._scan_path = None
        self._scan_name = None
        self._scan_title = None

        self._spec_scan = None
        self._spec_command = None
        self._spec_macro = None
        self._spec_args = None
        self._spec_scan_npts = None
        self._spec_scan_data = None
        self._spec_positioner_values = None

        self._detector_data_path = None

        if (isinstance(self, FMBRotationScanParser) and scan_number > 1
                and not self._previous_scan):
            scanparser = FMBRotationScanParser(
                spec_file_name, scan_number-1, previous_scan=True)
            if (scanparser.spec_macro in ('rams4_step_ome', 'rams4_fly_ome')
                    and len(scanparser.spec_args) == 5):
                self._rams4_args = scanparser.spec_args

    def __repr_(self):
        return (f'{self.__class__.__name__}'
                f'({self.spec_file_name}, {self.scan_number}) '
                f'-- {self.spec_command}')

    @property
    def spec_file(self):
        # NB This FileSpec instance is not stored as a private
        # attribute because it cannot be pickled (and therefore could
        # cause problems for parallel code that uses ScanParsers).
        return filespec(self.spec_file_name)

    @property
    def scan_path(self):
        if self._scan_path is None:
            self._scan_path = self.get_scan_path()
        return self._scan_path

    @property
    def scan_name(self):
        if self._scan_name is None:
            self._scan_name = self.get_scan_name()
        return self._scan_name

    @property
    def scan_title(self):
        if self._scan_title is None:
            self._scan_title = self.get_scan_title()
        return self._scan_title

    @property
    def spec_scan(self):
        if self._spec_scan is None:
            self._spec_scan = self.get_spec_scan()
        return self._spec_scan

    @property
    def spec_command(self):
        if self._spec_command is None:
            self._spec_command = self.get_spec_command()
        return self._spec_command

    @property
    def spec_macro(self):
        if self._spec_macro is None:
            self._spec_macro = self.get_spec_macro()
        return self._spec_macro

    @property
    def spec_args(self):
        if self._spec_args is None:
            self._spec_args = self.get_spec_args()
        return self._spec_args

    @property
    def spec_scan_npts(self):
        if self._spec_scan_npts is None:
            self._spec_scan_npts = self.get_spec_scan_npts()
        return self._spec_scan_npts

    @property
    def spec_scan_data(self):
        if self._spec_scan_data is None:
            self._spec_scan_data = self.get_spec_scan_data()
        return self._spec_scan_data

    @property
    def spec_positioner_values(self):
        if self._spec_positioner_values is None:
            self._spec_positioner_values = self.get_spec_positioner_values()
        return self._spec_positioner_values

    @property
    def detector_data_path(self):
        if self._detector_data_path is None:
            self._detector_data_path = self.get_detector_data_path()
        return self._detector_data_path

    def get_scan_path(self):
        """Return the name of the directory containining the SPEC file
        for this scan.

        :rtype: str
        """
        return os.path.dirname(self.spec_file_name)

    def get_scan_name(self):
        """Return the name of this SPEC scan (not unique to scans
        within a single spec file).

        :rtype: str
        """
        raise NotImplementedError

    def get_scan_title(self):
        """Return the title of this spec scan (unique to each scan
        within a spec file).

        :rtype: str
        """
        raise NotImplementedError

    def get_spec_scan(self):
        """Return the `pyspec.file.spec.Scan` object parsed from the
        spec file and scan number provided to the constructor.

        :rtype: pyspec.file.spec.Scan
        """
        return self.spec_file.getScanByNumber(self.scan_number)

    def get_spec_command(self):
        """Return the string command of this SPEC scan.

        :rtype: str
        """
        return self.spec_scan.command

    def get_spec_macro(self):
        """Return the macro used in this scan's SPEC command.

        :rtype: str
        """
        return self.spec_command.split()[0]

    def get_spec_args(self):
        """Return a list of the arguments provided to the macro for
        this SPEC scan.

        :rtype: list[str]
        """
        return self.spec_command.split()[1:]

    def get_spec_scan_npts(self):
        """Return the number of points collected in this SPEC scan

        :rtype: int
        """
        raise NotImplementedError

    def get_spec_scan_data(self):
        """Return a dictionary of all the counter data collected by
        this SPEC scan.

        :rtype: dict[str, numpy.ndarray]
        """
        return dict(zip(self.spec_scan.labels, self.spec_scan.data.T))

    def get_spec_positioner_values(self):
        """Return a dictionary of all the SPEC positioner values
        recorded by SPEC just before the scan began.

        :rtype: dict[str,str]
        """
        try:
            positioner_values = dict(self.spec_scan.motor_positions)
            names = list(positioner_values.keys())
            mnemonics = self.spec_scan.motors
        except Exception as e:
            raise ValueError(f'Error {e}')
        if mnemonics is not None:
            for name,mnemonic in zip(names,mnemonics):
                if name != mnemonic:
                    positioner_values[mnemonic] = positioner_values[name]
        return positioner_values

    def get_detector_data_path(self):
        """Return the name of the directory containing detector data
        collected by this scan.

        :rtype: str
        """
        raise NotImplementedError

    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        """Return the name of the file containing detector data
        collected at a certain step of this scan.

        :param detector_prefix: the prefix used in filenames for the
            detector
        :type detector_prefix: str
        :param scan_step_index: the index of the point in this scan
            whose detector file name should be returned.
        :type scan_step_index: int
        :rtype: str
        """
        raise NotImplementedError

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        """Return the detector data collected at a certain step of
        this scan.

        :param detector_prefix: the prefix used in filenames for the
            detector
        :type detector_prefix: str
        :param scan_step_index: the index of the point in this scan
            whose detector data should be returned.
        :type scan_step_index: int
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

    def get_spec_positioner_value(self, positioner_name):
        """Return the value of a spec positioner recorded before this
        scan began.

        :param positioner_name: the name or mnemonic of a SPEC motor
            whose position should be returned.
        :raises KeyError: if `positioner_name` is not the name or
            mnemonic of a SPEC motor recorded for this scan.
        :raises ValueError: if the recorded string value of the
            positioner in the SPEC file cannot be converted to a
            float.
        :rtype: float
        """
        try:
            positioner_value = self.spec_positioner_values[positioner_name]
            positioner_value = float(positioner_value)
        except KeyError:
            raise KeyError(f'{self.scan_title}: motor {positioner_name} '
                           'not found for this scan')
        except ValueError:
            raise ValueError(f'{self.scan_title}: could not convert value of'
                             f' {positioner_name} to float: '
                             f'{positioner_value}')
        return positioner_value


class FMBScanParser(ScanParser):
    """Partial implementation of a class representing a SPEC scan
    collected at FMB.
    """

    def get_scan_name(self):
        return os.path.basename(self.spec_file.abspath)

    def get_scan_title(self):
        return f'{self.scan_name}_{self.scan_number:03d}'


class SMBScanParser(ScanParser):
    """Partial implementation of a class representing a SPEC scan
    collected at SMB or FAST.
    """

    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)

        self._pars = None
        self._par_file_pattern = f'*-*-{self.scan_name}'
        self._par_file = None

    def get_scan_name(self):
        return os.path.basename(self.scan_path)

    def get_scan_title(self):
        return f'{self.scan_name}_{self.scan_number}'

    @property
    def pars(self):
        if self._pars is None:
            self._pars = self.get_pars()
        return self._pars

    def get_pars(self):
        """Return a dictionary of values recorded in the .par file
        associated with this SPEC scan.

        :rtype: dict[str,object]
        """
        # JSON file holds titles for columns in the par file
        json_files = fnmatch_filter(
            os.listdir(self.scan_path),
            f'{self._par_file_pattern}.json')
        if not json_files:
            json_files = fnmatch_filter(
                os.listdir(self.scan_path),
                f'*.json')
        if len(json_files) != 1:
            raise RuntimeError(f'{self.scan_title}: cannot find the '
                               '.json file to decode the .par file')
        with open(os.path.join(self.scan_path, json_files[0])) as json_file:
            par_file_cols = load(json_file)
        try:
            par_col_names = list(par_file_cols.values())
            scann_val_idx = par_col_names.index('SCAN_N')
            scann_col_idx = int(list(par_file_cols.keys())[scann_val_idx])
        except:
            raise RuntimeError(f'{self.scan_title}: cannot find scan pars '
                               'without a "SCAN_N" column in the par file')

        if getattr(self, '_par_file'):
            par_file = self._par_file
        else:
            par_files = fnmatch_filter(
                os.listdir(self.scan_path),
                f'{self._par_file_pattern}.par')
            if len(par_files) != 1:
                raise RuntimeError(f'{self.scan_title}: cannot find the .par '
                                   'file for this scan directory')
            par_file = os.path.join(self.scan_path, par_files[0])
            self._par_file = par_file
        par_dict = None
        with open(par_file) as f:
            par_reader = reader(f, delimiter=' ')
            for row in par_reader:
                if len(row) == len(par_col_names):
                    row_scann = int(row[scann_col_idx])
                    if row_scann == self.scan_number:
                        par_dict = {}
                        for par_col_idx,par_col_name in par_file_cols.items():
                            # Convert the string par value from the
                            # file to an int or float, if possible.
                            par_value = row[int(par_col_idx)]
                            try:
                                par_value = int(par_value)
                            except ValueError:
                                try:
                                    par_value = float(par_value)
                                except:
                                    pass
                            par_dict[par_col_name] = par_value
        if par_dict is None:
            raise RuntimeError(f'{self.scan_title}: could not find scan pars '
                               f'for scan number {self.scan_number}')
        return par_dict

    def get_counter_gain(self, counter_name):
        """Return the gain of a counter as recorded in the user lines
        of a scan in a SPEC file converted to nA/V.

        :param counter_name: the name of the counter
        :type counter_name: str
        :rtype: str
        """
        counter_gain = None
        for comment in self.spec_scan.comments + self.spec_scan.user_lines:
            match = re.search(
                f'{counter_name} gain: '  # start of counter gain comments
                '(?P<gain_value>\d+) '  # gain numerical value
                '(?P<unit_prefix>[m|u|n])A/V',  # gain units
                comment)
            if match:
                unit_prefix = match['unit_prefix']
                gain_scalar = 1 if unit_prefix == 'n' \
                    else 1e3 if unit_prefix == 'u' else 1e6
                counter_gain = f'{float(match["gain_value"])*gain_scalar} nA/V'
                break

        if counter_gain is None:
            raise RuntimeError(f'{self.scan_title}: could not get gain for '
                               f'counter {counter_name}')
        return counter_gain


class LinearScanParser(ScanParser):
    """Partial implementation of a class representing a typical line
    or mesh scan in SPEC.
    """
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)

        self._spec_scan_motor_mnes = None
        self._spec_scan_motor_vals = None
        self._spec_scan_motor_vals_relative = None
        self._spec_scan_shape = None
        self._spec_scan_dwell = None

    @property
    def spec_scan_motor_mnes(self):
        if self._spec_scan_motor_mnes is None:
            self._spec_scan_motor_mnes = self.get_spec_scan_motor_mnes()
        return self._spec_scan_motor_mnes

    @property
    def spec_scan_motor_vals(self):
        if self._spec_scan_motor_vals is None:
            self._spec_scan_motor_vals = self.get_spec_scan_motor_vals(
                relative=False)
        return self._spec_scan_motor_vals

    @property
    def spec_scan_motor_vals_relative(self):
        if self._spec_scan_motor_vals_relative is None:
            self._spec_scan_motor_vals_relative = \
                self.get_spec_scan_motor_vals(relative=True)
        return self._spec_scan_motor_vals_relative

    @property
    def spec_scan_shape(self):
        if self._spec_scan_shape is None:
            self._spec_scan_shape = self.get_spec_scan_shape()
        return self._spec_scan_shape

    @property
    def spec_scan_dwell(self):
        if self._spec_scan_dwell is None:
            self._spec_scan_dwell = self.get_spec_scan_dwell()
        return self._spec_scan_dwell

    def get_spec_scan_motor_mnes(self):
        """Return the mnemonics of the SPEC motor(s) provided to the
        macro for this scan. If there is more than one motor scanned
        (in a "flymesh" scan, for example), the order of motors in the
        returned tuple will go from the fastest moving motor first to
        the slowest moving motor last.

        :rtype: tuple
        """
        raise NotImplementedError

    def get_spec_scan_motor_vals(self, relative=False):
        """Return the values visited by each of the scanned motors. If
        there is more than one motor scanned (in a "flymesh" scan, for
        example), the order of motor values in the returned tuple will
        go from the fastest moving motor's values first to the slowest
        moving motor's values last.

        :param relative: If `True`, return scanned motor positions
            *relative* to the scanned motors' positions before the scan
            started, defaults to False.
        :type relative: bool, optional
        :rtype: tuple
        """
        raise NotImplementedError

    def get_spec_scan_shape(self):
        """Return the number of points visited by each of the scanned
        motors. If there is more than one motor scanned (in a
        "flymesh" scan, for example), the order of number of motor
        values in the returned tuple will go from the number of points
        visited by the fastest moving motor first to the the number of
        points visited by the slowest moving motor last.

        :rtype: tuple
        """
        raise NotImplementedError

    def get_spec_scan_dwell(self):
        """Return the dwell time for each point in the scan as it
        appears in the command string.

        :rtype: float
        """
        raise NotImplementedError

    def get_spec_scan_npts(self):
        """Return the number of points collected in this SPEC scan.

        :rtype: int
        """
        return np.prod(self.spec_scan_shape)

    def get_scan_step(self, scan_step_index:int):
        """Return the index of each motor coordinate corresponding to
        the index of a single point in the scan. If there is more than
        one motor scanned (in a "flymesh" scan, for example), the
        order of indices in the returned tuple will go from the index
        of the value of the fastest moving motor first to the index of
        the value of the slowest moving motor last.

        :param scan_step_index: the index of a single point in the
            scan.
        :type scan_step_index: int
        :rtype: tuple
        """
        scan_steps = np.ndindex(self.spec_scan_shape[::-1])
        i = 0
        while i <= scan_step_index:
            scan_step = next(scan_steps)
            i += 1
        return scan_step

    def get_scan_step_index(self, scan_step:tuple):
        """Return the index of a single scan point corresponding to a
        tuple of indices for each scanned motor coordinate.

        :param scan_step: a tuple of the indices of each scanned motor
            coordinate. If there is more than one motor scanned (in a
            "flymesh" scan, for example), the order of indices should
            go from the index of the value of the fastest moving motor
            first to the index of the value of the slowest moving
            motor last.
        :type scan_step: tuple
        :trype: int
        """
        scan_steps = np.ndindex(self.spec_scan_shape[::-1])
        scan_step_found = False
        scan_step_index = -1
        while not scan_step_found:
            next_scan_step = next(scan_steps)
            scan_step_index += 1
            if next_scan_step == scan_step:
                scan_step_found = True
                break
        return scan_step_index


class FMBLinearScanParser(LinearScanParser, FMBScanParser):
    """Partial implementation of a class representing a typical line
    or mesh scan in SPEC collected at FMB.
    """

    def get_spec_scan_motor_mnes(self):
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            # Fast motor
            m1_mne = self.spec_args[0]
            # Slow motor
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                m2_mne_i = 4
            else:
                m2_mne_i = 5
            m2_mne = self.spec_args[m2_mne_i]
            return (m1_mne, m2_mne)
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            return (self.spec_args[0],)
        if self.spec_macro in ('tseries', 'loopscan'):
            return ('Time',)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_motor_vals(self, relative=False):
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            # Fast motor
            m1_start = float(self.spec_args[1])
            m1_end = float(self.spec_args[2])
            m1_npt = int(self.spec_args[3]) + 1
            fast_mot_vals = np.linspace(m1_start, m1_end, m1_npt)
            # Slow motor
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                m2_start_i = 5
                m2_end_i = 6
                m2_nint_i = 7
            else:
                m2_start_i = 6
                m2_end_i = 7
                m2_nint_i = 8
            m2_start = float(self.spec_args[m2_start_i])
            m2_end = float(self.spec_args[m2_end_i])
            m2_npt = int(self.spec_args[m2_nint_i]) + 1
            slow_mot_vals = np.linspace(m2_start, m2_end, m2_npt)
            if relative:
                fast_mot_vals -= self.get_spec_positioner_value(
                    self.spec_scan_motor_mnes[0])
                slow_mot_vals -= self.get_spec_positioner_value(
                    self.spec_scan_motor_mnes[1])
            return (fast_mot_vals, slow_mot_vals)
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            mot_vals = np.linspace(float(self.spec_args[1]),
                                   float(self.spec_args[2]),
                                   int(self.spec_args[3])+1)
            if relative:
                mot_vals -= self.get_spec_positioner_value(
                    self.spec_scan_motor_mnes[0])
            return (mot_vals,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return (self.spec_scan.data[:,0],)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_shape(self):
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            # Fast motor
            fast_mot_npts = int(self.spec_args[3]) + 1
            # Slow motor
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                m2_nint_i = 7
            else:
                m2_nint_i = 8
            slow_mot_npts = int(self.spec_args[m2_nint_i]) + 1
            return (fast_mot_npts, slow_mot_npts)
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            mot_npts = int(self.spec_args[3]) + 1
            return (mot_npts,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return (len(np.array(self.spec_scan.data[:,0])),)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan shape '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_dwell(self):
        if self.macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                dwell = float(self.spec_args[8])
            return dwell
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            return float(self.spec_args[4])
        if self.spec_macro in ('tseries', 'loopscan'):
            return float(self.spec_args[1])
        raise RuntimeError(f'{self.scan_title}: cannot determine dwell for '
                           f'scans of type {self.spec_macro}')

    def get_detector_data_path(self):
        return os.path.join(self.scan_path, self.scan_title)


@cache
def list_fmb_saxswaxs_detector_files(detector_data_path, detector_prefix):
    """Return a sorted list of all data files for the given detector
    in the given directory. This function is cached to improve
    performace for carrying our full FAMB SAXS/WAXS data-processing
    workflows.

    :param detector_data_path: directory in which to look for detector
        data files
    :type detector_data_path: str
    :param detector_prefix: detector name to list files for
    :type detector_prefix: str
    :return: list of detector filenames
    :rtype: list[str]
    """
    return sorted(
        [f for f in os.listdir(detector_data_path)
        if detector_prefix in f
        and not f.endswith('.log')])


class FMBGIWAXSScanParser(FMBLinearScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical GIWAXS setup at FMB.
    """

    def get_scan_title(self):
        return f'{self.scan_name}_{self.scan_number:03d}'

    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        file_name = f'{self.scan_name}_{detector_prefix}_' \
                    f'{self.scan_number:03d}_{scan_step[0]:03d}.tiff'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        raise RuntimeError(f'{self.scan_title}: could not find detector image '
                           f'file for detector {detector_prefix} scan step '
                           f'({scan_step_index})')

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        detector_file = self.get_detector_data_file(
            detector_prefix, scan_step_index)
        with TiffFile(detector_file) as tiff_file:
            detector_data = tiff_file.asarray()
        return detector_data


class FMBSAXSWAXSScanParser(FMBLinearScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical SAXS/WAXS setup at FMB.
    """

    def get_scan_title(self):
        return f'{self.scan_name}_{self.scan_number:03d}'

    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        detector_files = list_fmb_saxswaxs_detector_files(
            self.detector_data_path, detector_prefix)
        if len(detector_files) == self.spec_scan_npts:
            return os.path.join(
                self.detector_data_path, detector_files[scan_step_index])
        else:
            scan_step = self.get_scan_step(scan_step_index)
            for f in detector_files:
                filename, _ = os.path.splitext(f)
                file_indices = tuple(
                    [int(i) for i in \
                     filename.split('_')[-len(self.spec_scan_shape):]])
                if file_indices == scan_step:
                    return os.path.join(self.detector_data_path, f)
            raise RuntimeError(
                'Could not find a matching detector data file for detector '
                + f'{detector_prefix} at scan step index {scan_step_index}')

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        import fabio
        image_file = self.get_detector_data_file(detector_prefix,
                                                 scan_step_index)
        with fabio.open(image_file) as det_file:
            image_data = det_file.data
        return image_data


class FMBXRFScanParser(FMBLinearScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical XRF setup at FMB.
    """

    def get_scan_title(self):
        return f'{self.scan_name}_scan{self.scan_number}'

    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        file_name = f'scan{self.scan_number}_{scan_step[0]:03d}.hdf5'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        raise RuntimeError(f'{self.scan_title}: could not find detector image '
                           f'file for detector {detector_prefix} scan step '
                           f'({scan_step_index})')

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        # Third party modules
        from h5py import File

        detector_file = self.get_detector_data_file(
            detector_prefix, scan_step_index)
        scan_step = self.get_scan_step(scan_step_index)
        with File(detector_file) as h5_file:
            detector_data = \
                h5_file['/entry/instrument/detector/data'][scan_step[0]]
        return detector_data


class SMBLinearScanParser(LinearScanParser, SMBScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical powder diffraction setup at SMB.
    """

    def get_spec_scan_dwell(self):
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                dwell = float(self.spec_args[8])
            return dwell
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            return float(self.spec_args[4])
        if self.spec_macro == 'tseries':
            return float(self.spec_args[1])
        if self.spec_macro == 'wbslew_scan':
            return float(self.spec_args[3])
        raise RuntimeError(f'{self.scan_title}: cannot determine dwell time '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_motor_mnes(self):
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            # Fast motor
            m1_mne = self.spec_args[0]
            # Slow motor
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                m2_mne_i = 4
            else:
                m2_mne_i = 5
            m2_mne = self.spec_args[m2_mne_i]
            return (m1_mne, m2_mne)
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            return (self.spec_args[0],)
        if self.spec_macro in ('tseries', 'loopscan'):
            return ('Time',)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_motor_vals(self, relative=False):
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            # Fast motor
            m1_start = float(self.spec_args[1])
            m1_end = float(self.spec_args[2])
            m1_npt = int(self.spec_args[3]) + 1
            fast_mot_vals = np.linspace(m1_start, m1_end, m1_npt)
            # Slow motor
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                m2_start_i = 5
                m2_end_i = 6
                m2_nint_i = 7
            else:
                m2_start_i = 6
                m2_end_i = 7
                m2_nint_i = 8
            m2_start = float(self.spec_args[m2_start_i])
            m2_end = float(self.spec_args[m2_end_i])
            m2_npt = int(self.spec_args[m2_nint_i]) + 1
            slow_mot_vals = np.linspace(m2_start, m2_end, m2_npt)
            if relative:
                fast_mot_vals -= float(self.spec_positioner_values[
                    self.spec_scan_motor_mnes[0]])
                slow_mot_vals -= float(self.spec_positioner_values[
                    self.spec_scan_motor_mnes[1]])
            return (fast_mot_vals, slow_mot_vals)
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            mot_vals = np.linspace(float(self.spec_args[1]),
                                   float(self.spec_args[2]),
                                   int(self.spec_args[3])+1)
            if relative:
                mot_vals -= self.get_spec_positioner_value(
                    self.spec_scan_motor_mnes[0])
            return (mot_vals,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return (self.spec_scan.data[:,0],)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_shape(self):
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            # Fast motor
            fast_mot_npts = int(self.spec_args[3]) + 1
            # Slow motor
            try:
                # Try post-summer-2022 format
                dwell = float(self.spec_args[4])
            except:
                # Accommodate pre-summer-2022 format
                m2_nint_i = 7
            else:
                m2_nint_i = 8
            slow_mot_npts = int(self.spec_args[m2_nint_i]) + 1
            return (fast_mot_npts, slow_mot_npts)
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            mot_npts = int(self.spec_args[3])+1
            return (mot_npts,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return (len(np.array(self.spec_scan.data[:,0])),)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan shape '
                           f'for scans of type {self.spec_macro}')

    def get_detector_data_path(self):
        return os.path.join(self.scan_path, str(self.scan_number))

    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        if len(scan_step) == 1:
            scan_step = (0, *scan_step)
        file_name_pattern = (f'{detector_prefix}_'
                             f'{self.scan_name}_*_'
                             f'{scan_step[0]}_data_'
                             f'{(scan_step[1]+1):06d}.h5')
        file_name_matches = fnmatch_filter(
            os.listdir(self.detector_data_path),
            file_name_pattern)
        if len(file_name_matches) == 1:
            return os.path.join(self.detector_data_path, file_name_matches[0])
        raise RuntimeError(f'{self.scan_title}: could not find detector image '
                           f'file for detector {detector_prefix} scan step '
                           f'({scan_step_index})')

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        # Third party modules
        from h5py import File

        image_file = self.get_detector_data_file(
            detector_prefix, scan_step_index)
        with File(image_file) as h5_file:
            image_data = h5_file['/entry/data/data'][0]
        return image_data


class RotationScanParser(ScanParser):
    """Partial implementation of a class representing a rotation
    scan.
    """

    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
        self._starting_image_index = None
        self._starting_image_offset = None

    @property
    def starting_image_index(self):
        if self._starting_image_index is None:
            self._starting_image_index = self.get_starting_image_index()
        return self._starting_image_index

    @property
    def starting_image_offset(self):
        if self._starting_image_offset is None:
            self._starting_image_offset = self.get_starting_image_offset()
        return self._starting_image_offset

    def get_starting_image_index(self):
        """Return the first frame of the detector data collected by
        this scan from the index of the first frame of detector data
        collected by this scan.

        :rtype: int
        """
        raise NotImplementedError

    def get_starting_image_offset(self):
        """Return the offset of the index of the first "good" frame of
        detector data collected by this scan from the index of the
        first frame of detector data collected by this scan.

        :rtype: int
        """
        raise NotImplementedError


class FMBRotationScanParser(RotationScanParser, FMBScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical tomography setup at FMB.
    """
    def __init__(
            self, spec_file_name, scan_number, previous_scan=False):
        self._previous_scan = previous_scan
        super().__init__(spec_file_name, scan_number)

    def get_spec_scan_data(self):
        spec_scan_data = super().get_spec_scan_data()
        if hasattr(self, '_rams4_args'):
            spec_scan_data['theta'] = np.linspace(
                float(self._rams4_args[0]), float(self._rams4_args[1]),
                int(self._rams4_args[2]) + 1)
        return spec_scan_data

    def get_spec_scan_npts(self):
        if hasattr(self, '_rams4_args'):
            return int(self._rams4_args[2]) + 1
        if self.spec_macro == 'flyscan':
            if len(self.spec_args) == 2:
                return int(self.spec_args[0]) + 1
            if len(self.spec_args) == 5:
                return int(self.spec_args[3]) + 1
            raise RuntimeError(f'{self.scan_title}: cannot obtain number of '
                               f'points from {self.spec_macro} with arguments '
                               f'{self.spec_args}')
#        if self.spec_macro == 'ascan':
#            if len(self.spec_args) == 5:
#                return int(self.spec_args[3])
#            raise RuntimeError(f'{self.scan_title}: cannot obtain number of '
#                               f'points from {self.spec_macro} with arguments '
#                               f'{self.spec_args}')
        raise RuntimeError(f'{self.scan_title}: cannot determine rotation '
                           f' angles for scans of type {self.spec_macro}')

    def get_starting_image_index(self):
        return 0

    def get_starting_image_offset(self):
        if hasattr(self, '_rams4_args'):
            return int(self.spec_args[0]) - self.spec_scan_npts
        if self.spec_macro == 'flyscan':
            return 1
        raise RuntimeError(f'{self.scan_title}: cannot determine starting '
                           f'image offset for scans of type {self.spec_macro}')

    def get_detector_data_path(self):
        return self.scan_path

    def get_detector_data_file(self, detector_prefix):
        prefix = detector_prefix.upper()
        file_name = f'{self.scan_name}_{prefix}_{self.scan_number:03d}.h5'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        raise RuntimeError(f'{self.scan_title}: could not find detector image '
                           f'file for detector {detector_prefix}')

    def get_all_detector_data_in_file(
            self, detector_prefix, scan_step_index=None):
        # Third party modules
        from h5py import File

        detector_file = self.get_detector_data_file(detector_prefix)
        with File(detector_file) as h5_file:
            if scan_step_index is None:
                detector_data = h5_file['/entry/instrument/detector/data'][
                    self.starting_image_offset:]
            elif isinstance(scan_step_index, int):
                detector_data = h5_file['/entry/instrument/detector/data'][
                    self.starting_image_offset+scan_step_index]
            elif (isinstance(scan_step_index, (list, tuple))
                    and len(scan_step_index) == 2):
                detector_data = h5_file['/entry/instrument/detector/data'][
                    self.starting_image_offset+scan_step_index[0]:
                    self.starting_image_offset+scan_step_index[1]]
            else:
                raise ValueError('Invalid parameter scan_step_index '
                                 f'({scan_step_index})')
        return detector_data

    def get_detector_data(self, detector_prefix, scan_step_index=None):
        try:
            # Detector files in h5 format
            detector_data = self.get_all_detector_data_in_file(
                detector_prefix, scan_step_index)
        except:
            # Detector files in tiff format
            if scan_step_index is None:
                detector_data = []
                for index in range(self.spec_scan_npts):
                    detector_data.append(
                        self.get_detector_data(detector_prefix, index))
                detector_data = np.asarray(detector_data)
            elif isinstance(scan_step_index, int):
                image_file = self._get_detector_tiff_file(
                    detector_prefix, scan_step_index)
                if image_file is None:
                    detector_data = None
                else:
                    with TiffFile(image_file) as tiff_file:
                        detector_data = tiff_file.asarray()
            elif (isinstance(scan_step_index, (list, tuple))
                    and len(scan_step_index) == 2):
                detector_data = []
                for index in range(scan_step_index[0], scan_step_index[1]):
                    detector_data.append(
                        self.get_detector_data(detector_prefix, index))
                detector_data = np.asarray(detector_data)
            else:
                raise ValueError('Invalid parameter scan_step_index '
                                 f'({scan_step_index})')
        return detector_data

    def _get_detector_tiff_file(self, detector_prefix, scan_step_index):
        file_name_full = f'{self.spec_file_name}_{detector_prefix.upper()}_' \
            f'{self.scan_number:03d}_' \
            f'{self.starting_image_offset+scan_step_index:03d}.tiff'
        if os.path.isfile(file_name_full):
            return file_name_full
        return None
#        raise RuntimeError(f'{self.scan_title}: could not find detector image '
#                           f'file for scan step ({scan_step_index})')


class SMBRotationScanParser(RotationScanParser, SMBScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical tomography setup at SMB.
    """

    def __init__(self, spec_file_name, scan_number, par_file=None):
        self._scan_type = None
        super().__init__(spec_file_name, scan_number)

        self._katefix = 0  # RV remove when no longer needed
        self._par_file_pattern = f'id*-*tomo*-{self.scan_name}'
        if par_file is not None:
            self._par_file = par_file

    @property
    def scan_type(self):
        if self._scan_type is None:
            self._scan_type = self.get_scan_type()
        return self._scan_type

    def get_spec_scan_data(self):
        spec_scan_data = super().get_spec_scan_data()
        spec_scan_data['theta'] = np.linspace(
            float(self.pars['ome_start_real']),
            float(self.pars['ome_end_real']), int(self.pars['nframes_real']))
        return spec_scan_data

    def get_spec_scan_npts(self):
        return int(self.pars['nframes_real'])

    def get_scan_type(self):
        scan_type = self.pars.get(
            'tomo_type', self.pars.get(
            'tomotype', self.pars.get('scan_type', None)))
        if scan_type is None:
            raise RuntimeError(
                f'{self.scan_title}: cannot determine the scan_type')
        return scan_type

    def get_starting_image_index(self):
        try:
            junkstart = int(self.pars['junkstart'])
            #RV temp fix for error in par files at Kate's beamline
            #Remove this and self._katefix when no longer needed
            file_name = f'nf_{junkstart:06d}.tif'
            file_name_full = os.path.join(self.detector_data_path, file_name)
            if not os.path.isfile(file_name_full):
                self._katefix = min([
                    int(re.findall(r'\d+', f)[0])
                        for f in os.listdir(self.detector_data_path)
                        if re.match(r'nf_\d+\.tif', f)])
            return junkstart
            #return int(self.pars['junkstart'])
        except:
            raise RuntimeError(f'{self.scan_title}: cannot determine first '
                               'detector image index')

    def get_starting_image_offset(self):
        try:
            return (int(self.pars['goodstart'])-self.starting_image_index)
        except:
            raise RuntimeError(f'{self.scan_title}: cannot determine index '
                               'offset of first good detector image')

    def get_detector_data_path(self):
        return os.path.join(self.scan_path, str(self.scan_number), 'nf')

    def get_detector_data_file(self, scan_step_index:int):
        index = self.starting_image_index + self.starting_image_offset \
            + scan_step_index
        file_name = f'nf_{index:06d}.tif'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        #RV temp fix for error in par files at Kate's beamline
        #Remove this and self._katefix when no longer needed
        index += self._katefix
        file_name = f'nf_{index:06d}.tif'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        raise RuntimeError(f'{self.scan_title}: could not find detector image '
                           f'file ({file_name_full}) for scan step '
                           f'({scan_step_index})')

    def get_detector_data(self, detector_prefix, scan_step_index=None):
        if scan_step_index is None:
            detector_data = []
            for index in range(self.spec_scan_npts):
                detector_data.append(
                    self.get_detector_data(detector_prefix, index))
            detector_data = np.asarray(detector_data)
        elif isinstance(scan_step_index, int):
            image_file = self.get_detector_data_file(scan_step_index)
            with TiffFile(image_file) as tiff_file:
                detector_data = tiff_file.asarray()
        elif (isinstance(scan_step_index, (list, tuple))
                and len(scan_step_index) == 2):
            detector_data = []
            for index in range(scan_step_index[0], scan_step_index[1]):
                detector_data.append(
                    self.get_detector_data(detector_prefix, index))
            detector_data = np.asarray(detector_data)
        else:
            raise ValueError('Invalid parameter scan_step_index '
                             f'({scan_step_index})')
        return detector_data


class MCAScanParser(ScanParser):
    """Partial implementation of a class representing a scan taken
    while collecting SPEC MCA data.
    """

    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)

        self._detector_num_bins = None

    def get_detector_num_bins(self, detector_prefix):
        """Return the number of bins for the detector with the given
        prefix.

        :param detector_prefix: the detector prefix as used in SPEC
            MCA data files
        :type detector_prefix: str
        :rtype: int
        """
        raise NotImplementedError


class SMBMCAScanParser(MCAScanParser, SMBLinearScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical EDD setup at SMB or FAST.
    """
    detector_data_formats = ('spec', 'h5')

    def __init__(self, spec_file_name, scan_number, detector_data_format=None):
        """Constructor for SMBMCAScnaParser.

        :param spec_file: Path to scan's SPEC file
        :type spec_file: str
        :param scan_number: Number of the SPEC scan
        :type scan_number: int
        :param detector_data_format: Format of the MCA data collected,
            defaults to None
        :type detector_data_format: Optional[Literal["spec", "h5"]]
        """
        super().__init__(spec_file_name, scan_number)

        self.detector_data_format = None
        if detector_data_format is None:
            self.init_detector_data_format()
        else:
            if detector_data_format.lower() in self.detector_data_formats:
                self.detector_data_format = detector_data_format.lower()
            else:
                raise ValueError(
                    'Unrecognized value for detector_data_format: '
                    f'{detector_data_format}. Allowed values are: '
                    ', '.join(self.detector_data_formats))

    def get_spec_scan_motor_vals(self, relative=True):
        if not relative:
            # The scanned motor's recorded position in the spec.log
            # file's "#P" lines does not always give the right offset
            # to use to obtain absolute motor postions from relative
            # motor positions (or relative from actual). Sometimes,
            # the labx/y/z/ometotal value from the scan's .par file is
            # the quantity for the offset that _should_ be used, but
            # there is currently no consistent way to determine when
            # to use the labx/y/z/ometotal .par file value and when to
            # use the spec file "#P" lines value. Because the relative
            # motor values are the only ones currently used in EDD
            # workflows, obtain them from relevant values available in
            # the .par file, and defer implementation for absolute
            # motor postions to later.
            return super().get_spec_scan_motor_vals(relative=True)
        if self.spec_macro in ('flymesh', 'mesh', 'flydmesh', 'dmesh'):
            # Fast motor
            mot_vals_axis0 = np.linspace(self.pars['fly_axis0_start'],
                                         self.pars['fly_axis0_end'],
                                         self.pars['fly_axis0_npts'])
            # Slow motor
            mot_vals_axis1 = np.linspace(self.pars['fly_axis1_start'],
                                         self.pars['fly_axis1_end'],
                                         self.pars['fly_axis1_npts'])
            return (mot_vals_axis0, mot_vals_axis1)
        if self.spec_macro in ('flyscan', 'ascan', 'flydscan', 'dscan'):
            mot_vals = np.linspace(self.pars['fly_axis0_start'],
                                   self.pars['fly_axis0_end'],
                                   self.pars['fly_axis0_npts'])
            return (mot_vals,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return (self.spec_scan.data[:,0],)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def init_detector_data_format(self):
        """Determine and set a value for the instance variable
        `detector_data_format` based on the presence / absence of
        detector data files of different formats conventionally
        associated with this scan. Also set the corresponding
        appropriate value for `_detector_data_path`.
        """
        try:
            self._detector_data_path = self.scan_path
            self.get_detector_data_file_spec()
        except OSError:
            try:
                self._detector_data_path = os.path.join(
                    self.scan_path, str(self.scan_number), 'edd')
                self.get_detector_data_files_h5()
            except OSError:
                raise RuntimeError(f'{self.scan_title}: Unable to determine '
                                   'detector data format')
            else:
                self.detector_data_format = 'h5'
        else:
            self.detector_data_format = 'spec'

    def get_detector_data_path(self):
        raise NotImplementedError

    def get_detector_num_bins(self):
        if self.detector_data_format == 'spec':
            return self.get_detector_num_bins_spec()
        if self.detector_data_format == 'h5':
            return self.get_detector_num_bins_h5()

    def get_detector_num_bins_spec(self):
        with open(self.get_detector_data_file_spec()) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#@CHANN'):
                try:
                    line_prefix, number_saved, first_saved, last_saved, \
                        reduction_coef = line.split()
                    return int(number_saved)
                except:
                    continue
        raise RuntimeError(f'{self.scan_title}: could not find num_bins')

    def get_detector_num_bins_h5(self):
        # Third party modules
        from h5py import File

        with File(self.get_detector_data_files_h5()[0]) as h5_file:
            dset_shape = h5_file['/entry/data/data'].shape
        return dset_shape[-1]

    def get_detector_data_file(self):
        raise NotImplementedError

    def get_detector_data_file_spec(self):
        """Return the filename (full absolute path) to the file
        containing spec-formatted MCA data for this scan.
        """
        filename = f'spec.log.scan{self.scan_number}.mca1.mca'
        filename_full = os.path.join(self.detector_data_path, filename)
        if os.path.isfile(filename_full):
            return filename_full
        raise OSError(f'Unable to find detector file {filename_full}')

    def get_detector_data_files_h5(self):
        """Return the filenames (full absolute paths) to the files
        containing h5-formatted MCA data for this scan.
        """
        filenames = sorted(
            [f for f in os.listdir(self.detector_data_path)
             if f.endswith('.hdf5')])
        filenames_full = []
        for filename in filenames:
            filenames_full.append(
                os.path.join(self.detector_data_path, filename))
            if not os.path.isfile(filenames_full[-1]):
                raise OSError(
                    f'Unable to find detector file {filenames_full[-1]}')
        return filenames_full

    def get_all_detector_data(self, detector=None):
        """Return a 3D array of all MCA spectra collected by the
        detector elements during the scan.

        :param detector: For detector data collected in SPEC format,
            this is the detector prefix as it appears in the spec MCA
            data file. For detector data collected in H5 format, this
            is a list of MCA detector element indices to return data.
            for. Defaults to `None` (return data for all detector
            elements, invalid for SPEC format detector data).
        :type detector: Union[str, list[int]), optional
        :returns: The MCA spectra
        :rtype: numpy.ndarray
        """
        # Local modules
        from CHAP.utils.general import is_int_series

        if self.detector_data_format == 'spec':
            raise RuntimeError(
                'SMBMCAScanParser.get_all_detector_data not updated/tested')
            if detector is None:
                raise TypeError('Missing required detector parameter')
            if not isinstance(detector, str):
                raise TypeError('Invalid detector parameter ({detector})')
            return self.get_all_detector_data_spec(detector)
        if self.detector_data_format == 'h5':
            if (detector is not None
                    and not is_int_series(detector, ge=0, log=False)):
                raise TypeError('Invalid detector parameter ({detector})')
            return self.get_all_detector_data_h5(detector)

    def get_all_detector_data_spec(self, detector_prefix):
        """Return a 2D array of all MCA spectra collected by a
        detector in the spec MCA file format during the scan.

        :param detector_prefix: Detector name at is appears in the
            spec MCA file.
        :type detector_prefix: str
        :returns: 2D array of MCA spectra
        :rtype: numpy.ndarray
        """
        # This should be easy with pyspec, but there are bugs in
        # pyspec for MCA data.....  or is the 'bug' from a nonstandard
        # implementation of some macro on our end?  According to spec
        # manual and pyspec code, mca data should always begin w/ '@A'
        # In example scans, it begins with '@{detector_prefix}'
        # instead
        data = []

        with open(self.get_detector_data_file_spec()) as detector_file:
            lines = [line.strip("\\\n") for line in detector_file.readlines()]

        num_bins = self.get_detector_num_bins()

        counter = 0
        for line in lines:
            a = line.split()

            if len(a) > 0:
                if a[0] == ("@"+detector_prefix):
                    counter = 1
                    spectrum = np.zeros(num_bins)
            if counter == 1:
                b = np.array(a[1:]).astype('uint16')
                spectrum[(counter-1) * 25:((counter-1) * 25 + 25)] = b
                counter = counter + 1
            elif counter > 1 and counter <= (np.floor(num_bins / 25.)):
                b = np.array(a).astype('uint16')
                spectrum[(counter-1) * 25:((counter-1) * 25 + 25)] = b
                counter = counter + 1
            elif counter == (np.ceil(num_bins/25.)):
                b = np.array(a).astype('uint16')
                spectrum[(counter-1) * 25:
                         ((counter-1) * 25 + (np.mod(num_bins, 25)))] = b
                data.append(spectrum)
                counter = 0

        return np.array(data)

    def get_all_detector_data_h5(self, detector_indices=None):
        """Return a 3D array of all MCA spectra collected by the
        detector elements during the scan in the h5 file format.

        :param detector_indices: A list of MCA detector element
            indices to return data for, default to `None` (return
            data for all detector elements).
        :type detector_indices: list[int], optional
        :returns: The MCA spectra
        :rtype: numpy.ndarray
        """
        detector_data = []
        for detector_file in self.get_detector_data_files_h5():
            data = self.get_all_mca_data_h5(detector_file)
            assert data.shape[0] == self.spec_scan_shape[0]
            if detector_indices is None:
                detector_data.append(data)
            else:
                detector_data.append(data[:,detector_indices,:])
        if len(self.spec_scan_shape) == 1:
            assert len(detector_data) == 1
            return np.asarray(detector_data[0])
        assert len(detector_data) == self.spec_scan_shape[1]
        return np.vstack(tuple(detector_data))

    def get_all_mca_data_h5(self, filename):
        """Return a 3D array of all MCA spectra collected by the
        detector elements for a single h5 data file.

        :param filename: Name of the MCA h5 data file
        :type filename: str
        :returns: 3D array of MCA spectra where the first index is the
            scan step, the second index is the detector element index,
            and the third index is channel energy bin.
        :rtype: numpy.ndarray
        """
        # Third partry modules
        from h5py import File

        with File(filename) as h5_file:
            data = h5_file['/entry/data/data'][:]

        # Prior to 2023-12-12, there was an issue where the XPS23 detector
        # was capturing one or two frames of all 0s at the start of the
        # dataset in every hdf5 file. In both cases, there is only ONE
        # extra frame of data relative to the number of frames that should
        # be there (based on the number of points in the spec scan). If
        # one frame of all 0s is present: skip it and deliver only the
        # real data. If two frames of all 0s are present: detector data
        # will be missing for the LAST step in the scan. Skip the first
        # two frames of all 0s in the hdf5 dataset, then add a frame of
        # fake data (all 0-s) to the end of that real data so that the
        # number of detector data frames matches the number of points in
        # the spec scan.
        check_zeros_before = 1702357200
        file_mtime = os.path.getmtime(filename)
        if file_mtime <= check_zeros_before:
            if not np.any(data[0]):
                # If present, remove first frame of blank data
                print('Warning: removing blank first frame of detector data')
                data = data[1:]
                if not np.any(data[0]):
                    # If present, shift second frame of blank data to the
                    # end
                    print('Warning: shifting second frame of blank detector data '
                          + 'to the end of the scan')
                    data = np.concatenate((data[1:], np.asarray([data[0]])))

        return data

    def get_detector_data(self, detector=None, scan_step_index=None):
        """Return a single MCA spectrum for the detector indicated.

        :param detector: For detector data collected in SPEC format,
            this is the detector prefix as it appears in the spec MCA
            data file. For detector data collected in H5 format, this
            is a list of MCA detector element indices to return data.
            for. Defaults to `None` (return data for all detector
            elements, invalid for SPEC format detector data).
        :type detector: Union[str, list[int]), optional
        :param scan_step_index: Index of the scan step to return the
            spectrum from.
        :type scan_step_index: int
        :returns: A single MCA spectrum
        :rtype: numpy.ndarray
        """
        detector_data = self.get_all_detector_data(detector)
        if scan_step_index is None:
            return detector_data
        # FIX scan_step_index needs testing
        raise RuntimeError('scan_step_index needs testing/updating')
        return detector_data[scan_step_index]

