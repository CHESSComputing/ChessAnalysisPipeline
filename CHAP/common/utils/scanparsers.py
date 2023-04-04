#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# system modules
import csv
import fnmatch
from functools import cache
import json
import os
import re

# necessary for the base class, ScanParser:
import numpy as np
from pyspec.file.spec import FileSpec            

class ScanParser(object):
    def __init__(self,
                 spec_file_name:str,
                 scan_number:int):

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
        return(f'{self.__class__.__name__}({self.spec_file_name}, {self.scan_number}) -- {self.spec_command}')
        
    @property
    def spec_file(self):
        # NB This FileSpec instance is not stored as a private attribute because
        # it cannot be pickled (and therefore could cause problems for
        # parallel code that uses ScanParsers).
        return(FileSpec(self.spec_file_name))
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
        return(os.path.dirname(self.spec_file_name))
    def get_scan_name(self):
        return(None)
    def get_scan_title(self):
        return(None)
    def get_spec_scan(self):
        return(self.spec_file.getScanByNumber(self.scan_number))
    def get_spec_command(self):
        return(self.spec_scan.command)
    def get_spec_macro(self):
        return(self.spec_command.split()[0])
    def get_spec_args(self):
        return(self.spec_command.split()[1:])
    def get_spec_scan_npts(self):
        raise(NotImplementedError)
    def get_spec_scan_data(self):
        return(dict(zip(self.spec_scan.labels, self.spec_scan.data.T)))
    def get_spec_positioner_values(self):
        positioner_values = dict(self.spec_scan.motor_positions)
        names = list(positioner_values.keys())
        mnemonics = self.spec_scan.motors
        if mnemonics is not None:
            for name,mnemonic in zip(names,mnemonics):
                if name != mnemonic:
                    positioner_values[mnemonic] = positioner_values[name]
        return(positioner_values)
    def get_detector_data_path(self):
        raise(NotImplementedError)
    
    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        raise(NotImplementedError)
    def get_detector_data(self, detector_prefix, scan_step_index:int):
        '''
        Return a np.ndarray of detector data.

        :param detector_prefix: The detector's name in any data files, often
        the EPICS macro $(P).
        :type detector_substring: str

        :param scan_step_index: The index of the scan step for which detector
            data will be returned.
        :type scan_step_index: int

        :return: The detector data
        :rtype: np.ndarray
        '''
        raise(NotImplementedError)

    def get_spec_positioner_value(self, positioner_name):
        try:
            positioner_value = self.spec_positioner_values[positioner_name]
            positioner_value = float(positioner_value)
            return(positioner_value)
        except KeyError:
            raise(KeyError(f'{self.scan_title}: motor {positioner_name} not found for this scan'))
        except ValueError:
            raise(ValueError(f'{self.scan_title}: ccould not convert value of {positioner_name} to float: {positioner_value}'))


class FMBScanParser(ScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
    def get_scan_name(self):
        return(os.path.basename(self.spec_file.abspath))
    def get_scan_title(self):
        return(f'{self.scan_name}_{self.scan_number:03d}')



class SMBScanParser(ScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
        
        self._pars = None # purpose: store values found in the .par file as a dictionary
        self.par_file_pattern = f'*-*-{self.scan_name}'
        
    def get_scan_name(self):
        return(os.path.basename(self.scan_path))
    def get_scan_title(self):
        return(f'{self.scan_name}_{self.scan_number}')
    
    @property
    def pars(self):
        if self._pars is None:
            self._pars = self.get_pars()
        return(self._pars)
        
    def get_pars(self):
        # JSON file holds titles for columns in the par file
        json_files = fnmatch.filter(os.listdir(self.scan_path), f'{self.par_file_pattern}.json')
        if not len(json_files) == 1:
            raise(RuntimeError(f'{self.scan_title}: cannot find the .json file to decode the .par file'))
        with open(os.path.join(self.scan_path, json_files[0])) as json_file:
            par_file_cols = json.load(json_file)
        try:
            par_col_names = list(par_file_cols.values())
            scann_val_idx = par_col_names.index('SCAN_N')
            scann_col_idx = int(list(par_file_cols.keys())[scann_val_idx])
        except:
            raise(RuntimeError(f'{self.scan_title}: cannot find scan pars without a "SCAN_N" column in the par file'))
        
        par_files = fnmatch.filter(os.listdir(self.scan_path), f'{self.par_file_pattern}.par')
        if not len(par_files) == 1:
            raise(RuntimeError(f'{self.scan_title}: cannot find the .par file for this scan directory'))
        with open(os.path.join(self.scan_path, par_files[0])) as par_file:
            par_reader = csv.reader(par_file, delimiter=' ')
            for row in par_reader:
                if len(row) == len(par_col_names):
                    row_scann = int(row[scann_col_idx])
                    if row_scann == self.scan_number:
                        par_dict = {}
                        for par_col_idx,par_col_name in par_file_cols.items():
                            # Convert the string par value from the file to an int or float, if possible.
                            par_value = row[int(par_col_idx)]
                            try:
                                par_value = int(par_value)
                            except ValueError:
                                try:
                                    par_value = float(par_value)
                                except:
                                    pass
                            par_dict[par_col_name] = par_value
                        return(par_dict)
        raise(RuntimeError(f'{self.scan_title}: could not find scan pars for scan number {self.scan_number}'))
            
    def get_counter_gain(self, counter_name):
        for comment in self.spec_scan.comments:
            match = re.search(f'{counter_name} gain: (?P<gain_value>\d+) (?P<unit_prefix>[m|u|n])A/V', comment)
            if match:
                unit_prefix = match['unit_prefix']
                gain_scalar = 1 if unit_prefix == 'n' else 1e3 if unit_prefix == 'u' else 1e6
                counter_gain = f'{float(match["gain_value"])*gain_scalar} nA/V'
                return(counter_gain)
        raise(RuntimeError(f'{self.scan_title}: could not get gain for counter {counter_name}'))


class LinearScanParser(ScanParser):
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
        return(self._spec_scan_dwell)

    def get_spec_scan_motor_names(self):
        raise(NotImplementedError)
    def get_spec_scan_motor_vals(self):
        raise(NotImplementedError)
    def get_spec_scan_shape(self):
        raise(NotImplementedError)
    def get_spec_scan_npts(self):
        return(np.prod(self.spec_scan_shape))
    def get_scan_step(self, scan_step_index:int):
        scan_steps = np.ndindex(self.spec_scan_shape[::-1])
        i = 0
        while i <= scan_step_index:
            scan_step = next(scan_steps)
            i += 1
        return(scan_step)
    def get_scan_step_index(self, scan_step:tuple):
        scan_steps = np.ndindex(self.spec_scan_shape[::-1])
        scan_step_found = False
        scan_step_index = -1
        while not scan_step_found:
            next_scan_step = next(scan_steps)
            scan_step_index += 1
            if next_scan_step == scan_step:
                scan_step_found = True
                break
        return(scan_step_index)


class FMBLinearScanParser(LinearScanParser, FMBScanParser):
    def __init__(self, spec_file_name, scan_number): 
        super().__init__(spec_file_name, scan_number)
        
    def get_spec_scan_motor_mnes(self):
        if self.spec_macro == 'flymesh':
            return((self.spec_args[0], self.spec_args[5]))
        elif self.spec_macro == 'flyscan':
            return((self.spec_args[0],))
        elif self.spec_macro in ('tseries', 'loopscan'):
            return(('Time',))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine scan motors for scans of type {self.spec_macro}'))
    def get_spec_scan_motor_vals(self):
        if self.spec_macro == 'flymesh':
            fast_mot_vals = np.linspace(float(self.spec_args[1]), float(self.spec_args[2]), int(self.spec_args[3])+1)
            slow_mot_vals = np.linspace(float(self.spec_args[6]), float(self.spec_args[7]), int(self.spec_args[8])+1)
            return((fast_mot_vals, slow_mot_vals))
        elif self.spec_macro == 'flyscan':
            mot_vals = np.linspace(float(self.spec_args[1]), float(self.spec_args[2]), int(self.spec_args[3])+1)
            return((mot_vals,))
        elif self.spec_macro in ('tseries', 'loopscan'):
            return(self.spec_scan.data[:,0])
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine scan motors for scans of type {self.spec_macro}'))
    def get_spec_scan_shape(self):
        if self.spec_macro == 'flymesh':
            fast_mot_npts = int(self.spec_args[3])+1
            slow_mot_npts = int(self.spec_args[8])+1
            return((fast_mot_npts, slow_mot_npts))
        elif self.spec_macro == 'flyscan':
            mot_npts = int(self.spec_args[3])+1
            return((mot_npts,))
        elif self.spec_macro in ('tseries', 'loopscan'):
            return(len(np.array(self.spec_scan.data[:,0])))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine scan shape for scans of type {self.spec_macro}'))
    def get_spec_scan_dwell(self):
        if self.spec_macro in ('flymesh', 'flyscan'):
            return(float(self.spec_args[4]))
        elif self.spec_macro in ('tseries', 'loopscan'):
            return(float(self.spec_args[1]))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine dwell for scans of type {self.spec_macro}'))
    def get_detector_data_path(self):
        return(os.path.join(self.scan_path, self.scan_title))


class FMBSAXSWAXSScanParser(FMBLinearScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)

    def get_scan_title(self):
        return(f'{self.scan_name}_{self.scan_number:03d}')
    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        file_indices = [f'{scan_step[i]:03d}' for i in range(len(self.spec_scan_shape)) if self.spec_scan_shape[i] != 1]
        file_name = f'{self.scan_name}_{detector_prefix}_{self.scan_number:03d}_{"_".join(file_indices)}.tiff'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return(file_name_full)
        else:
            raise(RuntimeError(f'{self.scan_title}: could not find detector image file for detector {detector_prefix} scan step ({scan_step})'))
    def get_detector_data(self, detector_prefix, scan_step_index:int):
        from pyspec.file.tiff import TiffFile
        image_file = self.get_detector_data_file(detector_prefix, scan_step_index)
        with TiffFile(image_file) as tiff_file:
            image_data = tiff_file.asarray()
        return(image_data)


class FMBXRFScanParser(FMBLinearScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
    def get_scan_title(self):
        return(f'{self.scan_name}_scan{self.scan_number}')        
    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        file_name = f'scan{self.scan_number}_{scan_step[1]:03d}.hdf5'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return(file_name_full)
        else:
            raise(RuntimeError(f'{self.scan_title}: could not find detector image file for detector {detector_prefix} scan step ({scan_step_index})'))
    def get_detector_data(self, detector_prefix, scan_step_index:int):
        import h5py
        detector_file = self.get_detector_data_file(detector_prefix, scan_step_index)
        scan_step = self.get_scan_step(scan_step_index)
        with h5py.File(detector_file) as h5_file:
            detector_data = h5_file['/entry/instrument/detector/data'][scan_step[0]]
        return(detector_data) 


class SMBLinearScanParser(LinearScanParser, SMBScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)        
    def get_spec_scan_motor_mnes(self):
        if self.spec_macro == 'flymesh':
            return((self.spec_args[0], self.spec_args[5]))
        elif self.spec_macro == 'flyscan':
            return((self.spec_args[0],))
        elif self.spec_macro in ('tseries', 'loopscan'):
            return(('Time',))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine scan motors for scans of type {self.spec_macro}'))
    def get_spec_scan_motor_vals(self):
        if self.spec_macro == 'flymesh':
            fast_mot_vals = np.linspace(float(self.spec_args[1]), float(self.spec_args[2]), int(self.spec_args[3])+1)
            slow_mot_vals = np.linspace(float(self.spec_args[6]), float(self.spec_args[7]), int(self.spec_args[8])+1)
            return((fast_mot_vals, slow_mot_vals))
        elif self.spec_macro == 'flyscan':
            mot_vals = np.linspace(float(self.spec_args[1]), float(self.spec_args[2]), int(self.spec_args[3])+1)
            return((mot_vals,))
        elif self.spec_macro in ('tseries', 'loopscan'):
            return(self.spec_scan.data[:,0])
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine scan motors for scans of type {self.spec_macro}'))
    def get_spec_scan_shape(self):
        if self.spec_macro == 'flymesh':
            fast_mot_npts = int(self.spec_args[3])+1
            slow_mot_npts = int(self.spec_args[8])+1
            return((fast_mot_npts, slow_mot_npts))
        elif self.spec_macro == 'flyscan':
            mot_npts = int(self.spec_args[3])+1
            return((mot_npts,))
        elif self.spec_macro in ('tseries', 'loopscan'):
            return(len(np.array(self.spec_scan.data[:,0])))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine scan shape for scans of type {self.spec_macro}'))
    def get_spec_scan_dwell(self):
        if self.spec_macro == 'flymesh':
            return(float(self.spec_args[4]))
        elif self.spec_macro == 'flyscan':
            return(float(self.spec_args[-1]))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine dwell time for scans of type {self.spec_macro}'))
    def get_detector_data_path(self):
        return(os.path.join(self.scan_path, str(self.scan_number)))
    def get_detector_data_file(self, detector_prefix, scan_step_index:int):
        scan_step = self.get_scan_step(scan_step_index)
        if len(scan_step) == 1:
            scan_step = (0, *scan_step)
        file_name_pattern = f'{detector_prefix}_{self.scan_name}_*_{scan_step[0]}_data_{(scan_step[1]+1):06d}.h5'
        file_name_matches = fnmatch.filter(os.listdir(self.detector_data_path), file_name_pattern)
        if len(file_name_matches) == 1:
            return(os.path.join(self.detector_data_path, file_name_matches[0]))
        else:
            raise(RuntimeError(f'{self.scan_title}: could not find detector image file for detector {detector_prefix} scan step ({scan_step_index})'))
    def get_detector_data(self, detector_prefix, scan_step_index:int):
        import h5py
        image_file = self.get_detector_data_file(detector_prefix, scan_step_index)
        with h5py.File(image_file) as h5_file:
            image_data = h5_file['/entry/data/data'][0]
        return(image_data)


class RotationScanParser(ScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)

        self._scan_type = None
        self._theta_vals = None
        self._horizontal_shift = None
        self._vertical_shift = None
        self._starting_image_index = None
        self._starting_image_offset = None

    @property
    def scan_type(self):
        if self._scan_type is None:
            self._scan_type = self.get_scan_type()
        return(self._scan_type)
    @property
    def theta_vals(self):
        if self._theta_vals is None:
            self._theta_vals = self.get_theta_vals()
        return(self._theta_vals)
    @property
    def horizontal_shift(self):
        if self._horizontal_shift is None:
            self._horizontal_shift = self.get_horizontal_shift()
        return(self._horizontal_shift)
    @property
    def vertical_shift(self):
        if self._vertical_shift is None:
            self._vertical_shift = self.get_vertical_shift()
        return(self._vertical_shift)
    @property
    def starting_image_index(self):
        if self._starting_image_index is None:
            self._starting_image_index = self.get_starting_image_index()
        return(self._starting_image_index)
    @property
    def starting_image_offset(self):
        if self._starting_image_offset is None:
            self._starting_image_offset = self.get_starting_image_offset()
        return(self._starting_image_offset)
 
    def get_scan_type(self):
        return(None)
    def get_theta_vals(self):
        raise(NotImplementedError)
    def get_horizontal_shift(self):
        raise(NotImplementedError)
    def get_vertical_shift(self):
        raise(NotImplementedError)
    def get_starting_image_index(self):
        raise(NotImplementedError)
    def get_starting_image_offset(self):
        raise(NotImplementedError)
    def get_num_image(self, detector_prefix):
        raise(NotImplementedError)


class FMBRotationScanParser(RotationScanParser, FMBScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
    def get_theta_vals(self):
        if self.spec_macro == 'flyscan':
            if len(self.spec_args) == 2:
                # Flat field (dark or bright)
                return({'num': int(self.spec_args[0])})
            elif len(self.spec_args) == 5:
                return({'start': float(self.spec_args[1]), 'end': float(self.spec_args[2]),
                        'num': int(self.spec_args[3])+1})
            else:
                raise(RuntimeError(f'{self.scan_title}: cannot obtain theta values from '+
                        f'{self.spec_macro} with arguments {self.spec_args}'))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine scan motors for scans '+
                    f'of type {self.spec_macro}'))
    def get_horizontal_shift(self):
        return(0.0)
    def get_vertical_shift(self):
        return(float(self.get_spec_positioner_value('4C_samz')))
    def get_starting_image_index(self):
        return(0)
    def get_starting_image_offset(self):
        return(1)
    def get_num_image(self, detector_prefix):
        import h5py
        detector_file = self.get_detector_data_file(detector_prefix)
        with h5py.File(detector_file) as h5_file:
            num_image = h5_file['/entry/instrument/detector/data'].shape[0]
        return(num_image-self.starting_image_offset)
    def get_detector_data_path(self):
        return(self.scan_path)
    def get_detector_data_file(self, detector_prefix):
        prefix = detector_prefix.upper()
        file_name = f'{self.scan_name}_{prefix}_{self.scan_number:03d}.h5'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return(file_name_full)
        else:
            raise(RuntimeError(f'{self.scan_title}: could not find detector image file for '+
                    f'detector {detector_prefix}'))
    #@cache
    def get_all_detector_data_in_file(self, detector_prefix, scan_step_index=None):
        import h5py
        detector_file = self.get_detector_data_file(detector_prefix)
        with h5py.File(detector_file) as h5_file:
            if scan_step_index is None:
                detector_data = h5_file['/entry/instrument/detector/data'][
                        self.starting_image_index:]
            elif isinstance(scan_step_index, int):
                detector_data = h5_file['/entry/instrument/detector/data'][
                        self.starting_image_index+scan_step_index]
            elif isinstance(scan_step_index, (list, tuple)) and len(scan_step_index) == 2:
                detector_data = h5_file['/entry/instrument/detector/data'][
                        self.starting_image_index+scan_step_index[0]:
                        self.starting_image_index+scan_step_index[1]]
            else:
                raise(ValueError(f'Invalid parameter scan_step_index ({scan_step_index})'))
        return(detector_data)
    def get_detector_data(self, detector_prefix, scan_step_index=None):
        return(self.get_all_detector_data_in_file(detector_prefix, scan_step_index))


class SMBRotationScanParser(RotationScanParser, SMBScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
        self.par_file_pattern = f'id*-*tomo*-{self.scan_name}'
    def get_scan_type(self):
        try:
            return(self.pars['tomo_type'])
        except:
            try:
                return(self.pars['tomotype'])
            except:
                raise(RuntimeError(f'{self.scan_title}: cannot determine the scan_type'))
    def get_theta_vals(self):
        return({'start': float(self.pars['ome_start_real']),
                'end': float(self.pars['ome_end_real']), 'num': int(self.pars['nframes_real'])})
    def get_horizontal_shift(self):
        try:
            return(float(self.pars['rams4x']))
        except:
            try:
                return(float(self.pars['ramsx']))
            except:
                raise(RuntimeError(f'{self.scan_title}: cannot determine the horizontal shift'))
    def get_vertical_shift(self):
        try:
            return(float(self.pars['rams4z']))
        except:
            try:
                return(float(self.pars['ramsz']))
            except:
                raise(RuntimeError(f'{self.scan_title}: cannot determine the vertical shift'))
    def get_starting_image_index(self):
        try:
            return(int(self.pars['junkstart']))
        except:
            raise(RuntimeError(f'{self.scan_title}: cannot determine first detector image index'))
    def get_starting_image_offset(self):
        try:
            return(int(self.pars['goodstart'])-self.get_starting_image_index())
        except:
            raise(RuntimeError(f'{self.scan_title}: cannot determine index offset of first good '+
                    'detector image'))
    def get_num_image(self, detector_prefix=None):
        try:
            return(int(self.pars['nframes_real']))
#            indexRegex = re.compile(r'\d+')
#            # At this point only tiffs
#            path = self.get_detector_data_path()
#            files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
#                    f.endswith('.tif') and indexRegex.search(f)])
#            return(len(files)-self.starting_image_offset)
        except:
            raise(RuntimeError(f'{self.scan_title}: cannot determine the number of good '+
                    'detector images'))
    def get_detector_data_path(self):
        return(os.path.join(self.scan_path, str(self.scan_number), 'nf'))
    def get_detector_data_file(self, scan_step_index:int):
        file_name = f'nf_{self.starting_image_index+scan_step_index:06d}.tif'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return(file_name_full)
        else:
            raise(RuntimeError(f'{self.scan_title}: could not find detector image file for '+
                    f'scan step ({scan_step_index})'))
    def get_detector_data(self, detector_prefix, scan_step_index=None):
        if scan_step_index is None:
            detector_data = []
            for index in range(len(self.get_num_image(detector_prefix))):
                detector_data.append(self.get_detector_data(detector_prefix, index))
            detector_data = np.asarray(detector_data)
        elif isinstance(scan_step_index, int):
            image_file = self.get_detector_data_file(scan_step_index)
            from pyspec.file.tiff import TiffFile
            with TiffFile(image_file) as tiff_file:
                detector_data = tiff_file.asarray()
        elif isinstance(scan_step_index, (list, tuple)) and len(scan_step_index) == 2:
            detector_data = []
            for index in range(scan_step_index[0], scan_step_index[1]):
                detector_data.append(self.get_detector_data(detector_prefix, index))
            detector_data = np.asarray(detector_data)
        else:
            raise(ValueError(f'Invalid parameter scan_step_index ({scan_step_index})'))
        return(detector_data)


class MCAScanParser(ScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
        
        self._dwell_time = None
        self._detector_num_bins = None
        
    @property
    def dwell_time(self):
        if self._dwell_time is None:
            self._dwell_time = self.get_dwell_time()
        return(self._dwell_time)
    
    def get_dwell_time(self):
        raise(NotImplementedError)
    @cache
    def get_detector_num_bins(self, detector_prefix):
        raise(NotImplementedError)

class SMBMCAScanParser(MCAScanParser, SMBScanParser):
    def __init__(self, spec_file_name, scan_number):
        super().__init__(spec_file_name, scan_number)
            
    def get_spec_scan_npts(self):
        if self.spec_macro == 'tseries':
            return(1)
        elif self.spec_macro == 'ascan':
            return(int(self.spec_args[3]))
        elif self.spec_scan == 'wbslew_scan':
            return(1)
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine number of points for scans of type {self.spec_macro}'))

    def get_dwell_time(self):
        if self.spec_macro == 'tseries':
            return(float(self.spec_args[1]))
        elif self.spec_macro == 'ascan':
            return(float(self.spec_args[4]))
        elif self.spec_macro == 'wbslew_scan':
            return(float(self.spec_args[3]))
        else:
            raise(RuntimeError(f'{self.scan_title}: cannot determine dwell time for scans of type {self.spec_macro}'))

    def get_detector_num_bins(self, detector_prefix):
        with open(self.get_detector_file(detector_prefix)) as detector_file:
            lines = detector_file.readlines()
        for line in lines:
            if line.startswith('#@CHANN'):
                try:
                    line_prefix, number_saved, first_saved, last_saved, reduction_coef = line.split()
                    return(int(number_saved))
                except:
                    continue
        raise(RuntimeError(f'{self.scan_title}: could not find num_bins for detector {detector_prefix}'))
    
    def get_detector_data_path(self):
        return(self.scan_path)

    def get_detector_file(self, detector_prefix, scan_step_index:int=0):
        file_name = f'spec.log.scan{self.scan_number}.mca1.mca'
        file_name_full = os.path.join(self.detector_data_path, file_name)
        if os.path.isfile(file_name_full):
            return(file_name_full)
        else:
            raise(RuntimeError(f'{self.scan_title}: could not find detector image file'))

    @cache
    def get_all_detector_data(self, detector_prefix):
        # This should be easy with pyspec, but there are bugs in pyspec for MCA data.....
        # or is the 'bug' from a nonstandard implementation of some macro on our end?
        # According to spec manual and pyspec code, mca data should always begin w/ '@A'
        # In example scans, it begins with '@mca1' instead
        data = []
        
        with open(self.get_detector_file(detector_prefix)) as detector_file:
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
                spectrum[(counter-1)*25:((counter-1)*25+25)] = b
                counter = counter + 1
            elif counter > 1 and counter <= (np.floor(num_bins/25.)):
                b = np.array(a).astype('uint16')
                spectrum[(counter-1)*25:((counter-1)*25+25)] = b
                counter = counter+1
            elif counter == (np.ceil(num_bins/25.)):
                b = np.array(a).astype('uint16')
                spectrum[(counter-1)*25:((counter-1)*25+(np.mod(num_bins,25)))] = b
                data.append(spectrum)
                counter = 0

        return(data)

    def get_detector_data(self, detector_prefix, scan_step_index:int):
        detector_data = self.get_all_detector_data(detector_prefix)
        return(detector_data[scan_step_index])
