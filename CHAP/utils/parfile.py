"""Utilities for interacting with scans using an SMB-style .par file
as input
"""

import csv
import json
import os

class ParFile():
    """Representation of a .par file

    :ivar par_file: name of the .par file
    :type par_file: str
    :ivar json_file: name of the .json file containing the key for
        column names of the .par file
    :type json_file: str
    :ivar spec_file: name of the SPEC data file associated with this
        .par file
    :type spec_file: str
    :ivar column_names: list of the names of each column in the par file
    :type column_names: list[str]
    :ivar data: a 2D array of the data in this .par file. 0th index:
        row. 1st index: column
    :type data: list[list]
    """
    def __init__(self, par_file, scann_col_name='SCAN_N'):
        self.par_file = str(par_file)
        self.json_file = self.par_file.replace('.par', '.json')
        self.spec_file = os.path.join(
            os.path.dirname(self.par_file), 'spec.log')

        with open(self.json_file) as json_file:
            columns = json.load(json_file)
        self.column_names = [None] * len(columns)
        for i, name in columns.items():
            self.column_names[int(i)] = name

        self.data = []
        with open(self.par_file) as par_file:
            reader = csv.reader(par_file, delimiter=' ')
            for row in reader:
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
                self.data.append(row_data)

        self.scann_i = self.column_names.index(scann_col_name)
        self.scan_numbers = [data[self.scann_i] for data in self.data]

    def get_map(self, experiment_type, station, dim_columns):
        """Return a map configuration based on this par file.

        :param experiment_type: name of the experiment type for the
            map that this .par file represents
        :type experiment_type: Literal['SAXSWAXS', 'EDD', 'XRF', 'TOMO']
        :param station: name of the station at which the data were
            collected
        :type station: Literal['id1a3','id3a','id3b']
        :param dim_columns: list of dictionaries configuring the map's
            independent dimensions.
        :type dim_columns: list[dict[str, str]]
        :return: a map configuration
        :rtype: CHAP.common.models.map.MapConfig
        """
        from CHAP.common.models.map import MapConfig
        from CHAP.utils.scanparsers import SMBScanParser

        scanparser = SMBScanParser(self.spec_file, 1)
        map_config = {
            'title': scanparser.scan_name,
            'station': station, #scanparser.station,
            'experiment_type': experiment_type,
            'sample': {'name': scanparser.scan_name},
            'spec_scans': [
                {'spec_file': self.spec_file,
                 'scan_numbers': self.good_scan_numbers()}],
            'independent_dimensions': [
                {'label': col['label'],
                 'units': col['units'],
                 'name': col['name'],
                 'data_type': 'smb_par'}
                for col in dim_columns]
        }
        return MapConfig(**map_config)

    def good_scan_numbers(self, good_col_name='1/0'):
        """Return the numbers of scans marked with a "1" in the
        indicated "good" column of the .par file.
        
        :param good_col_name: the name of the "good" column of the par
            file, defaults to "1/0"
        :type good_col_name: str, optional
        :raises ValueError: if this .par file does not have a column
            with the same name as `good_col_name`
        :return: "good" scan numbers
        :rtype: list[int]
        """
        good_col_i = self.column_names.index(good_col_name)
        return [self.scan_numbers[i] for i in range(len(self.scan_numbers))
                if self.data[i][good_col_i] == 1]
        
    def get_values(self, column, scan_numbers=None):
        """Return values from a single column of the par file.

        :param column: the string name OR index of the column to return
            values for
        :type column: str or int
        :param scan_numbers: list of specific scan numbers to return
            values in the given column for (instead of the default
            behavior: return the entire column of values), defaults to
            None
        :type scan_numbers: list[int], optional
        :return: a list of values from a single column in the par file
        :rtype: list[object]
        """
        if isinstance(column, str):
            column_idx = self.column_names.index(column)
        elif isinstance(column, int):
            column_idx = column
        else:
            raise TypeError(f'column must be a str or int, not {type(column)}')

        column_data = [self.data[i][column_idx] for i in range(len(self.data))]
        if scan_numbers is not None:
            column_data = [column_data[self.scan_numbers.index(scan_n)] \
                           for scan_n in scan_numbers]
        return column_data
