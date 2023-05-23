#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# System modules
from csv import reader
from fnmatch import filter as fnmatch_filter
from json import load
import os
import re

# Third party modules
import numpy as np
from pyspec.file.spec import FileSpec
from pyspec.file.tiff import TiffFile


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

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'({self.spec_file_name}, {self.scan_number}) '
                f'-- {self.spec_command}')

    @property
    def spec_file(self):
        # NB This FileSpec instance is not stored as a private
        # attribute because it cannot be pickled (and therefore could
        # cause problems for parallel code that uses ScanParsers).
        return FileSpec(self.spec_file_name)

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
        positioner_values = dict(self.spec_scan.motor_positions)
        names = list(positioner_values.keys())
        mnemonics = self.spec_scan.motors
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

        par_files = fnmatch_filter(
            os.listdir(self.scan_path),
            f'{self._par_file_pattern}.par')
        if len(par_files) != 1:
            raise RuntimeError(f'{self.scan_title}: cannot find the .par '
                               'file for this scan directory')
        par_dict = None
        with open(os.path.join(self.scan_path, par_files[0])) as par_file:
            par_reader = reader(par_file, delimiter=' ')
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
                               'for scan number {self.scan_number}')
        return par_dict

    def get_counter_gain(self, counter_name):
        """Return the gain of a counter as recorded in the comments of
        a scan in a SPEC file converted to nA/V.

        :param counter_name: the name of the counter
        :type counter_name: str
        :rtype: str
        """
        counter_gain = None
        for comment in self.spec_scan.comments:
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
            self._spec_scan_motor_vals = self.get_spec_scan_motor_vals()
        return self._spec_scan_motor_vals

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

    def get_spec_scan_motor_vals(self):
        """Return the values visited by each of the scanned motors. If
        there is more than one motor scanned (in a "flymesh" scan, for
        example), the order of motor values in the returned tuple will
        go from the fastest moving motor's values first to the slowest
        moving motor's values last.

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
        if self.spec_macro == 'flymesh':
            return (self.spec_args[0], self.spec_args[5])
        if self.spec_macro in ('flyscan', 'ascan'):
            return (self.spec_args[0],)
        if self.spec_macro in ('tseries', 'loopscan'):
            return ('Time',)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_motor_vals(self):
        if self.spec_macro == 'flymesh':
            fast_mot_vals = np.linspace(float(self.spec_args[1]),
                                        float(self.spec_args[2]),
                                        int(self.spec_args[3])+1)
            slow_mot_vals = np.linspace(float(self.spec_args[6]),
                                        float(self.spec_args[7]),
                                        int(self.spec_args[8])+1)
            return (fast_mot_vals, slow_mot_vals)
        if self.spec_macro in ('flyscan', 'ascan'):
            mot_vals = np.linspace(float(self.spec_args[1]),
                                   float(self.spec_args[2]),
                                   int(self.spec_args[3])+1)
            return (mot_vals,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return self.spec_scan.data[:,0]
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_shape(self):
        if self.spec_macro == 'flymesh':
            fast_mot_npts = int(self.spec_args[3])+1
            slow_mot_npts = int(self.spec_args[8])+1
            return (fast_mot_npts, slow_mot_npts)
        if self.spec_macro in ('flyscan', 'ascan'):
            mot_npts = int(self.spec_args[3])+1
            return (mot_npts,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return len(np.array(self.spec_scan.data[:,0]))
        raise RuntimeError(f'{self.scan_title}: cannot determine scan shape '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_dwell(self):
        if self.spec_macro in ('flymesh', 'flyscan', 'ascan'):
            return float(self.spec_args[4])
        if self.spec_macro in ('tseries', 'loopscan'):
            return float(self.spec_args[1])
        raise RuntimeError(f'{self.scan_title}: cannot determine dwell for '
                           f'scans of type {self.spec_macro}')

    def get_detector_data_path(self):
        return os.path.join(self.scan_path, self.scan_title)


class FMBSAXSWAXSScanParser(FMBLinearScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical SAXS/WAXS setup at FMB.
    """

    def get_scan_title(self):
        return f'{self.scan_name}_{self.scan_number:03d}'

    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        file_indices = [f'{scan_step[i]:03d}'
                        for i in range(len(self.spec_scan_shape))
                        if self.spec_scan_shape[i] != 1]
        if len(file_indices) == 0:
            file_indices = ['000']
        file_name = f'{self.scan_name}_{detector_prefix}_' \
                    f'{self.scan_number:03d}_{"_".join(file_indices)}.tiff'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        raise RuntimeError(f'{self.scan_title}: could not find detector image '
                           f'file for detector {detector_prefix} scan step '
                           f'({scan_step})')

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        image_file = self.get_detector_data_file(detector_prefix,
                                                 scan_step_index)
        with TiffFile(image_file) as tiff_file:
            image_data = tiff_file.asarray()
        return image_data


class FMBXRFScanParser(FMBLinearScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical XRF setup at FMB.
    """

    def get_scan_title(self):
        return f'{self.scan_name}_scan{self.scan_number}'

    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        file_name = f'scan{self.scan_number}_{scan_step[1]:03d}.hdf5'
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
        if self.spec_macro in ('flymesh', 'flyscan', 'ascan'):
            return float(self.spec_args[4])
        if self.spec_macro == 'tseries':
            return float(self.spec_args[1])
        if self.spec_macro == 'wbslew_scan':
            return float(self.spec_args[3])
        raise RuntimeError(f'{self.scan_title}: cannot determine dwell time '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_motor_mnes(self):
        if self.spec_macro == 'flymesh':
            return (self.spec_args[0], self.spec_args[5])
        if self.spec_macro in ('flyscan', 'ascan'):
            return (self.spec_args[0],)
        if self.spec_macro in ('tseries', 'loopscan'):
            return ('Time',)
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_motor_vals(self):
        if self.spec_macro == 'flymesh':
            fast_mot_vals = np.linspace(float(self.spec_args[1]),
                                        float(self.spec_args[2]),
                                        int(self.spec_args[3])+1)
            slow_mot_vals = np.linspace(float(self.spec_args[6]),
                                        float(self.spec_args[7]),
                                        int(self.spec_args[8])+1)
            return (fast_mot_vals, slow_mot_vals)
        if self.spec_macro in ('flyscan', 'ascan'):
            mot_vals = np.linspace(float(self.spec_args[1]),
                                   float(self.spec_args[2]),
                                   int(self.spec_args[3])+1)
            return (mot_vals,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return self.spec_scan.data[:,0]
        raise RuntimeError(f'{self.scan_title}: cannot determine scan motors '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_shape(self):
        if self.spec_macro == 'flymesh':
            fast_mot_npts = int(self.spec_args[3])+1
            slow_mot_npts = int(self.spec_args[8])+1
            return (fast_mot_npts, slow_mot_npts)
        if self.spec_macro in ('flyscan', 'ascan'):
            mot_npts = int(self.spec_args[3])+1
            return (mot_npts,)
        if self.spec_macro in ('tseries', 'loopscan'):
            return len(np.array(self.spec_scan.data[:,0]))
        raise RuntimeError(f'{self.scan_title}: cannot determine scan shape '
                           f'for scans of type {self.spec_macro}')

    def get_spec_scan_dwell(self):
        if self.spec_macro == 'flymesh':
            return float(self.spec_args[4])
        if self.spec_macro in ('flyscan', 'ascan'):
            return float(self.spec_args[-1])
        raise RuntimeError(f'{self.scan_title}: cannot determine dwell time '
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

        self._scan_type = None
        self._rotation_angles= None
        self._horizontal_shift = None
        self._vertical_shift = None
        self._starting_image_index = None
        self._starting_image_offset = None

    @property
    def scan_type(self):
        if self._scan_type is None:
            self._scan_type = self.get_scan_type()
        return self._scan_type

    @property
    def rotation_angles(self):
        if self._rotation_angles is None:
            self._rotation_angles = self.get_rotation_angles()
        return self._rotation_angles

    @property
    def horizontal_shift(self):
        if self._horizontal_shift is None:
            self._horizontal_shift = self.get_horizontal_shift()
        return self._horizontal_shift

    @property
    def vertical_shift(self):
        if self._vertical_shift is None:
            self._vertical_shift = self.get_vertical_shift()
        return self._vertical_shift

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

    def get_spec_scan_npts(self):
        return len(self.rotation_angles)

    def get_scan_type(self):
        """Return a string identifier for the type of tomography data
        being collected by this scan: df1 (dark field), bf1 (bright
        field), or tf1 (sample tomography data).

        :rtype: typing.Literal['df1', 'bf1', 'tf1']
        """
        return None

    def get_rotation_angles(self):
        """Return the angular values visited by the rotating motor at
        each point in the scan.

        :rtype: np.array(float)"""
        raise NotImplementedError

    def get_horizontal_shift(self):
        """Return the value of the motor that shifts the sample in the
        +x direction (hutch frame). Useful when tomography scans are
        taken in a series of stacks when the sample is wider than the
        width of the beam.

        :rtype: float
        """
        raise NotImplementedError

    def get_vertical_shift(self):
        """Return the value of the motor that shifts the sample in the
        +z direction (hutch frame). Useful when tomography scans are
        taken in a series of stacks when the sample is taller than the
        height of the beam.

        :rtype: float
        """
        raise NotImplementedError

    def get_starting_image_index(self):
        """Return the index of the first frame of detector data
        collected by this scan.

        :rtype: int
        """
        raise NotImplementedError

    def get_starting_image_offset(self):
        """Return the offet of the index of the first "good" frame of
        detector data collected by this scan from the index of the
        first frame of detector data collected by this scan.

        :rtype: int
        """
        raise NotImplementedError


class FMBRotationScanParser(RotationScanParser, FMBScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical tomography setup at FMB.
    """

    def get_rotation_angles(self):
        if self.spec_macro == 'flyscan':
            if len(self.spec_args) == 2:
                # Flat field (dark or bright)
                return int(self.spec_args[0]) * [0]
            if (len(self.spec_args) == 5
                    and int(self.spec_args[3]) > 2*self.starting_image_offset):
                all_rotation_angles = np.linspace(
                    float(self.spec_args[1]), float(self.spec_args[2]),
                    int(self.spec_args[3])+1)
                return all_rotation_angles[
                    self.starting_image_offset:-1-self.starting_image_offset]
            raise RuntimeError(f'{self.scan_title}: cannot obtain rotation '
                               f'angles from {self.spec_macro} with '
                               f'arguments {self.spec_args}')
        raise RuntimeError(f'{self.scan_title}: cannot determine rotation '
                           f' angles for scans of type {self.spec_macro}')

    def get_horizontal_shift(self):
        try:
            horizontal_shift = float(self.get_spec_positioner_value('4C_samx'))
        except:
            try:
                horizontal_shift = float(
                    self.get_spec_positioner_value('GI_samx'))
            except:
                raise RuntimeError(
                    f'{self.scan_title}: cannot determine the horizontal shift')
        return horizontal_shift

    def get_vertical_shift(self):
        try:
            vertical_shift = float(self.get_spec_positioner_value('4C_samz'))
        except:
            try:
                vertical_shift = float(self.get_spec_positioner_value('GI_samz'))
            except:
                raise RuntimeError(
                    f'{self.scan_title}: cannot determine the vertical shift')
        return vertical_shift

    def get_starting_image_index(self):
        return 0

    def get_starting_image_offset(self):
        if len(self.spec_args) == 2:
            # Flat field (dark or bright)
            return 0
        if len(self.spec_args) == 5:
            return 1

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
                    self.starting_image_index:-1-self.starting_image_offset]
            elif isinstance(scan_step_index, int):
                detector_data = h5_file['/entry/instrument/detector/data'][
                    self.starting_image_index+scan_step_index]
            elif (isinstance(scan_step_index, (list, tuple))
                    and len(scan_step_index) == 2):
                detector_data = h5_file['/entry/instrument/detector/data'][
                    self.starting_image_index+scan_step_index[0]:
                    self.starting_image_index+scan_step_index[1]]
            else:
                raise ValueError('Invalid parameter scan_step_index '
                                 f'({scan_step_index})')
        return detector_data

    def get_detector_data(self, detector_prefix, scan_step_index=None):
        return self.get_all_detector_data_in_file(
            detector_prefix, scan_step_index)


class SMBRotationScanParser(RotationScanParser, SMBScanParser):
    """Concrete implementation of a class representing a scan taken
    with the typical tomography setup at SMB.
    """

    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)

        self._par_file_pattern = f'id*-*tomo*-{self.scan_name}'

    def get_scan_type(self):
        scan_type = self.pars.get(
            'tomo_type', self.pars.get('tomotype', None))
        if scan_type is None:
            raise RuntimeError(
                f'{self.scan_title}: cannot determine the scan_type')
        return scan_type

    def get_rotation_angles(self):
        return np.linspace(
            float(self.pars['ome_start_real']),
            float(self.pars['ome_end_real']), int(self.pars['nframes_real']))

    def get_horizontal_shift(self):
        horizontal_shift = self.pars.get(
            'rams4x', self.pars.get('ramsx', None))
        if horizontal_shift is None:
            raise RuntimeError(
                f'{self.scan_title}: cannot determine the horizontal shift')
        return horizontal_shift

    def get_vertical_shift(self):
        vertical_shift = self.pars.get(
            'rams4z', self.pars.get('ramsz', None))
        if vertical_shift is None:
            raise RuntimeError(
                f'{self.scan_title}: cannot determine the vertical shift')
        return vertical_shift

    def get_starting_image_index(self):
        try:
            return int(self.pars['junkstart'])
        except:
            raise RuntimeError(f'{self.scan_title}: cannot determine first '
                               'detector image index')

    def get_starting_image_offset(self):
        try:
            return (int(self.pars['goodstart'])
                    - self.get_starting_image_index())
        except:
            raise RuntimeError(f'{self.scan_title}: cannot determine index '
                               'offset of first good detector image')

    def get_detector_data_path(self):
        return os.path.join(self.scan_path, str(self.scan_number), 'nf')

    def get_detector_data_file(self, scan_step_index:int):
        file_name = f'nf_{self.starting_image_index+scan_step_index:06d}.tif'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        raise RuntimeError(f'{self.scan_title}: could not find detector image '
                           f'file for scan step ({scan_step_index})')

    def get_detector_data(self, detector_prefix, scan_step_index=None):
        if scan_step_index is None:
            detector_data = []
            for index in range(len(self.get_spec_scan_npts())):
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

    def get_detector_num_bins(self, detector_prefix):
        with open(self.get_detector_data_file(detector_prefix)) \
             as detector_file:
            lines = detector_file.readlines()
        for line in lines:
            if line.startswith('#@CHANN'):
                try:
                    line_prefix, number_saved, first_saved, last_saved, \
                        reduction_coef = line.split()
                    return int(number_saved)
                except:
                    continue
        raise RuntimeError(f'{self.scan_title}: could not find num_bins for '
                           f'detector {detector_prefix}')

    def get_detector_data_path(self):
        return self.scan_path

    def get_detector_data_file(self, detector_prefix, scan_step_index=0):
        file_name = f'spec.log.scan{self.scan_number}.mca1.mca'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return file_name_full
        raise RuntimeError(
            f'{self.scan_title}: could not find detector image file')

    def get_all_detector_data(self, detector_prefix):
        # This should be easy with pyspec, but there are bugs in
        # pyspec for MCA data.....  or is the 'bug' from a nonstandard
        # implementation of some macro on our end?  According to spec
        # manual and pyspec code, mca data should always begin w/ '@A'
        # In example scans, it begins with '@mca1' instead
        data = []

        with open(self.get_detector_data_file(detector_prefix)) \
                as detector_file:
            lines = [line.strip("\\\n") for line in detector_file.readlines()]

        num_bins = self.get_detector_num_bins(detector_prefix)

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

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        detector_data = self.get_all_detector_data(detector_prefix)
        return detector_data[scan_step_index]
