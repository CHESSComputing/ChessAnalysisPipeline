"""Utilities for interacting with scans using an SMB-style .par file
as input.
"""

# System modules
import csv
import os

# Third party modules
import json

class ParFile():
    """Representation of a .par file.

    :ivar par_file: Name of the .par file.
    :type par_file: str
    :ivar json_file: Name of the .json file containing the keys for
        the column names of the .par file.
    :type json_file: str
    :ivar spec_file: Name of the SPEC data file associated with this
        .par file.
    :type spec_file: str
    :ivar column_names: List of the names of each column in the par
        file.
    :type column_names: list[str]
    :ivar data: A 2D array of the data in this .par file. 0th index:
        row. 1st index: column
    :type data: list[list]
    """
    def __init__(self, par_file, scan_numbers=None, scann_col_name='SCAN_N'):
        # Local modules
        from CHAP.utils.general import (
            is_int_series,
            string_to_list,
        )

        self.par_file = str(par_file)
        self.json_file = self.par_file.replace('.par', '.json')
        self.spec_file = os.path.join(
            os.path.dirname(self.par_file), 'spec.log')

        with open(self.json_file) as json_file:
            columns = json.load(json_file)
        num_column = len(columns)
        self.column_names = [None] * num_column
        for i, name in columns.items():
            self.column_names[int(i)] = name

        self.data = []
        with open(self.par_file) as f:
            reader = csv.reader(f, delimiter=' ')
            for i, row in enumerate(reader):
                if len(row) == 0:
                    continue
                if row[0].startswith('#'):
                    continue
                row_data = []
                for value in row:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except:
                            pass
                    row_data.append(value)
                if len(row_data) != num_column:
                    raise ValueError(
                        'Mismatch between the number of columns in the json '
                        f'({num_column}) and line {i+1} of the par file '
                        f'({len(row_data)})')
                self.data.append(row_data)

        self.scann_i = self.column_names.index(scann_col_name)
        self.scan_numbers = [data[self.scann_i] for data in self.data]
        if scan_numbers is not None:
            if isinstance(scan_numbers, int):
                scan_numbers = [scan_numbers]
            elif isinstance(scan_numbers, str):
                scan_numbers = string_to_list(scan_numbers)
            if not is_int_series(scan_numbers, ge=0, log=False):
                raise TypeError(
                    f'Invalid scan_numbers parameter ({scan_numbers})')
            self.scan_numbers = [
                n for n in scan_numbers if n in self.scan_numbers]

    def get_map(
            self, experiment_type, station, par_dims, other_dims=None):
        """Return a map configuration based on this par file.

        :param experiment_type: Experiment type name for the map
            that this .par file represents.
        :type experiment_type:
            Literal['SAXSWAXS', 'EDD', 'XRF', 'TOMO']
        :param station: Station name at which the data were collected.
        :type station: Literal['id1a3','id3a','id3b']
        :param par_dims: List of dictionaries configuring the map's
            independent dimensions.
        :type par_dims: list[dict[str, str]]
        :param other_dims: List of other dimensions to include in
            the returned MapConfig's independednt_dimensions. Use this
            if each scans in thhis par ile captured more than one
            frame of data. Defaults to `None`
        :type other_dims: list[dict[str,str]], optional
        :return: The map configuration.
        :rtype: CHAP.common.models.map.MapConfig
        """
        # Third party modeuls
        # pylint: disable=import-error
        from chess_scanparsers import SMBScanParser
        # pylint: enable=import-error

        # Local modules
        from CHAP.common.models.map import MapConfig

        scanparser = SMBScanParser(self.spec_file, 1)
        good_scans = self.good_scan_numbers()
        if other_dims is None:
            other_dims = []
        map_config = {
            'title': scanparser.scan_name,
            'station': station, #scanparser.station,
            'experiment_type': experiment_type,
            'sample': {'name': scanparser.scan_name},
            'spec_scans': [
                {'spec_file': self.spec_file,
                 'scan_numbers': good_scans}],
            'independent_dimensions': [
                {'label': dim['label'],
                 'units': dim['units'],
                 'name': dim['name'],
                 'data_type': 'smb_par'}
                for dim in par_dims] + other_dims
        }
        return MapConfig(**map_config)

    def good_scan_numbers(self, good_col_name='1/0'):
        """Return the numbers of scans marked with a "1" in the
        indicated "good" column of the .par file.
        
        :param good_col_name: The name of the "good" column of the par
            file, defaults to "1/0"
        :type good_col_name: str, optional
        :raises ValueError: If this .par file does not have a column
            with the same name as `good_col_name`.
        :return: "good" scan numbers.
        :rtype: list[int]
        """
        good_col_i = self.column_names.index(good_col_name)
        return [self.scan_numbers[i] for i in range(len(self.scan_numbers))
                if self.data[i][good_col_i] == 1]

    def get_values(self, column, scan_numbers=None):
        """Return values from a single column of the par file.

        :param column: The string name OR index of the column to return
            values for.
        :type column: str or int
        :param scan_numbers: List of specific scan numbers to return
            values in the given column for (instead of the default
            behavior: return the entire column of values).
        :type scan_numbers: list[int], optional
        :raise:
            ValueError: Unavailable column name.
            TypeError: Illegal column name type.
        :return: A list of values from a single column in the par file.
        :rtype: list[object]
        """
        if isinstance(column, str):
            if column in self.column_names:
                column_idx = self.column_names.index(column)
#            elif column in ('dataset_id', 'scan_type'):
#                column_idx = None
            else:
                raise ValueError(f'Unavailable column name: {column} not in '
                                 f'{self.column_names}')
        elif isinstance(column, int):
            column_idx = column
        else:
            raise TypeError(f'column must be a str or int, not {type(column)}')

        if column_idx is None:
            column_data = [None]*len(self.data)
        else:
            column_data = [
                self.data[i][column_idx] for i in range(len(self.data))]
        if scan_numbers is not None:
            column_data = [column_data[self.scan_numbers.index(scan_n)] \
                           for scan_n in scan_numbers]
        return column_data

    def map_values(self, map_config, values):
        """Return a reshaped array of the 1D list `values` so that it
        matches up with the coordinates of `map_config`.

        :param map_config: The map configuration according to which
            values will be reshaped.
        :type map_config: MapConfig
        :param values: A 1D list of values to reshape.
        :type values: list or np.ndarray
        :return: Reshaped array of values.
        :rtype: np.ndarray
        """
        # Third party modules
        import numpy as np

        good_scans = self.good_scan_numbers()
        if len(values) != len(good_scans):
            raise ValueError('number of values provided ({len(values)}) does '
                             'not match the number of good scans in '
                             f'{self.par_file} ({len(good_scans)})')
        n_map_points = np.prod(map_config.shape)
        if len(values) != n_map_points:
            raise ValueError(
                f'Cannot reshape {len(values)} values into an array of shape '
                f'{map_config.shape}')

        map_values = np.empty(map_config.shape)
        for map_index in np.ndindex(map_config.shape):
            _, scan_number, _ = \
                map_config.get_scan_step_index(map_index)
            value_index = good_scans.index(scan_number)
            map_values[map_index] = values[value_index]
        return map_values
