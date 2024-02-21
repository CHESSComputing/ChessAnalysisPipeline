#!/usr/bin/env python
from CHAP.reader import Reader


class EddMapReader(Reader):
    """Reader for taking an EDD-style .par file and returning a
    `MapConfig` representing one of the datasets in the
    file. Independent dimensions are determined automatically, and a
    specific set of items to use for extra scalar datasets to include
    are hard-coded in."""
    def read(self, parfile, dataset_id):
        """Return a validated `MapConfig` object representing an EDD
        dataset.

        :param parfile: Name of the EDD-style .par file containing the
            dataset.
        :type parfile: str
        :param dataset_id: Number of the dataset in the .par file
            to return as a map.
        :type dataset_id: int
        :returns: Map configuration packaged with the appropriate
            value for 'schema'.
        :rtype: PipelineData
        """
        import numpy as np
        from CHAP.common.models.map import MapConfig
        from CHAP.pipeline import PipelineData
        from CHAP.utils.parfile import ParFile
        from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser

        parfile = ParFile(parfile)

        # Get list of scan numbers for the dataset
        dataset_ids = np.asarray(parfile.get_values('dataset_id'))
        dataset_rows_i = np.argwhere(
            np.where(
                np.asarray(dataset_ids) == dataset_id, 1, 0)).flatten()
        scan_nos = [parfile.data[i][parfile.scann_i] for i in dataset_rows_i\
                    if parfile.data[i][parfile.scann_i] in \
                        parfile.good_scan_numbers()]
        self.logger.debug(f'Scan numbers: {scan_nos}')
        spec_scans = [dict(spec_file=parfile.spec_file, scan_numbers=scan_nos)]

        # Get scan type for this dataset
        scan_types = parfile.get_values('scan_type', scan_numbers=scan_nos)
        if any([st != scan_types[0] for st in scan_types]):
            msg = 'Only one scan type per dataset is suported.'
            self.logger.error(msg)
            raise Exception(msg)
        scan_type = scan_types[0]
        self.logger.debug(f'Scan type: {scan_type}')

        # Based on scan type, get independent_dimensions for the map
        # Start by adding labx, laby, labz, and omega. Any "extra"
        # dimensions will be sqeezed out of the map later.
        independent_dimensions = [
            dict(label='labx', units='mm', data_type='smb_par',
                 name='labx'),
            dict(label='laby', units='mm', data_type='smb_par',
                 name='laby'),
            dict(label='labz', units='mm', data_type='smb_par',
                 name='labz'),
            dict(label='ometotal', units='degrees', data_type='smb_par',
                 name='ometotal')
        ]
        scalar_data = []
        attrs = {}
        if scan_type != 0:
            self.logger.warning(
                'Assuming all fly axes parameters are identical for all scans')
            attrs['fly_axis_labels'] = []
            axes_labels = {1: 'fly_labx', 2: 'fly_laby', 3: 'fly_labz',
                           4: 'fly_ometotal'}
            axes_units = {1: 'mm', 2: 'mm', 3: 'mm', 4: 'degrees'}
            axes_added = []
            scanparser = ScanParser(parfile.spec_file, scan_nos[0])
            def add_fly_axis(fly_axis_index):
                if fly_axis_index in axes_added:
                    return
                fly_axis_key = scanparser.pars[f'fly_axis{fly_axis_index}']
                independent_dimensions.append(dict(
                    label=axes_labels[fly_axis_key],
                    data_type='spec_motor',
                    units=axes_units[fly_axis_key],
                    name=scanparser.spec_scan_motor_mnes[fly_axis_index]))
                axes_added.append(fly_axis_index)
                attrs['fly_axis_labels'].append(axes_labels[fly_axis_key])
            add_fly_axis(0)
            if scan_type in (2, 3, 5):
                add_fly_axis(1)
            if scan_type == 5:
                scalar_data.append(dict(
                    label='bin_axis', units='n/a', data_type='smb_par',
                    name='bin_axis'))
                attrs['bin_axis_label'] = axes_labels[
                    scanparser.pars['bin_axis']].replace('fly_', '')

        # Add in the usual extra scalar data maps for EDD
        scalar_data.extend([
            dict(label='SCAN_N', units='n/a', data_type='smb_par',
                 name='SCAN_N'),
            dict(label='rsgap_size', units='mm', data_type='smb_par',
                 name='rsgap_size'),
            dict(label='x_effective', units='mm', data_type='smb_par',
                 name='x_effective'),
            dict(label='z_effective', units='mm', data_type='smb_par',
                 name='z_effective'),
        ])

        # Construct initial map config dictionary
        scanparser = ScanParser(parfile.spec_file, scan_nos[0])
        map_config_dict = dict(
            title=f'{scanparser.scan_name}_dataset{dataset_id}',
            station='id1a3',
            experiment_type='EDD',
            sample=dict(name=scanparser.scan_name),
            spec_scans=[
                dict(spec_file=parfile.spec_file, scan_numbers=scan_nos)],
            independent_dimensions=independent_dimensions,
            scalar_data=scalar_data,
            presample_intensity=dict(name='a3ic1', data_type='scan_column'),
            postsample_intensity=dict(name='diode', data_type='scan_column'),
            dwell_time_actual=dict(name='count_time', data_type='smb_par'),
            attrs=attrs
        )
        map_config = MapConfig(**map_config_dict)

        # Squeeze out extraneous independent dimensions (dimensions
        # along which data were taken at only one unique coordinate
        # value)
        while 1 in map_config.shape:
            remove_dim_index = map_config.shape[::-1].index(1)
            self.logger.debug(
                'Map dimensions: '
                + str([dim["label"] for dim in independent_dimensions]))
            self.logger.debug(f'Map shape: {map_config.shape}')
            self.logger.debug(
                'Sqeezing out independent dimension '
                + independent_dimensions[remove_dim_index]['label'])
            independent_dimensions.pop(remove_dim_index)
            map_config = MapConfig(**map_config_dict)
        self.logger.debug(
            'Map dimensions: '
            + str([dim["label"] for dim in independent_dimensions]))
        self.logger.debug(f'Map shape: {map_config.shape}')

        # Add lab coordinates to the map's scalar_data only if they
        # are NOT already one of the sqeezed map's
        # independent_dimensions.
        lab_dims = [
            dict(label='labx', units='mm', data_type='smb_par', name='labx'),
            dict(label='laby', units='mm', data_type='smb_par', name='laby'),
            dict(label='labz', units='mm', data_type='smb_par', name='labz'),
            dict(label='ometotal', units='degrees', data_type='smb_par',
                 name='ometotal')
        ]
        for dim in lab_dims:
            if dim not in independent_dimensions:
                scalar_data.append(dim)

        return map_config_dict


class ScanToMapReader(Reader):
    """Reader for turning a single SPEC scan into a MapConfig."""
    def read(self, spec_file, scan_number):
        """Return a dictionary representing a valid map configuration
        consisting of the single SPEC scan specified.

        :param spec_file: Name of the SPEC file.
        :type spec_file: str
        :param scan_number: Number of the SPEC scan.
        :type scan_number: int
        :returns: Map configuration dictionary
        :rtype: dict
        """
        from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser

        scanparser = ScanParser(spec_file, scan_number)

        if scanparser.spec_macro in ('tseries', 'loopscan') or \
           (scanparser.spec_macro == 'flyscan' and \
            not len(scanparser.spec_args) ==5):
            independent_dimensions = [
                {'label': 'Time',
                 'units': 'seconds',
                 'data_type': 'scan_column',
                 'name': 'Time'}]
        else:
            independent_dimensions = [
                {'label': mne,
                 'units': 'unknown units',
                 'data_type': 'spec_motor',
                 'name': mne}
                for mne in scanparser.spec_scan_motor_mnes]

        map_config_dict = dict(
            title=f'{scanparser.scan_name}_{scan_number:03d}',
            station='id1a3',
            experiment_type='EDD',
            sample=dict(name=scanparser.scan_name),
            spec_scans=[
                dict(spec_file=spec_file, scan_numbers=[scan_number])],
            independent_dimensions=independent_dimensions,
            presample_intensity=dict(name='a3ic1', data_type='scan_column'),
            postsample_intensity=dict(name='diode', data_type='scan_column'),
            dwell_time_actual=dict(name='count_time', data_type='smb_par')
        )

        return map_config_dict


if __name__ == '__main__':
    from CHAP.reader import main
    main()
