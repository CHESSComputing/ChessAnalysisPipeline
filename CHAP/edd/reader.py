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
        self.logger.debug(f'spec_file: {parfile.spec_file}')

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
            dwell_time_actual=dict(name='sec', data_type='scan_column'),
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
            dwell_time_actual=dict(name='sec', data_type='scan_column')
        )

        return map_config_dict


class SetupNXdataReader(Reader):
    """Reader for converting the SPEC input .txt file for EDD dataset
    collection to an approporiate input argument for
    `CHAP.common.SetupNXdataProcessor`.

    Example of use in a `Pipeline` configuration:
    ```yaml
    config:
      inputdir: /rawdata/samplename
      outputdir: /reduceddata/samplename
    pipeline:
      - edd.SetupNXdataReader:
          filename: SpecInput.txt
          dataset_id: 1
      - common.SetupNXdataProcessor:
          nxname: samplename_dataset_1
      - common.NexusWriter:
          filename: data.nxs
    ```
    """
    def read(self, filename, dataset_id):
        """Return a dictionary containing the `coords`, `signals`, and
        `attrs` arguments appropriate for use with
        `CHAP.common.SetupNXdataProcessor.process` to set up an
        initial `NXdata` object representing a complete and organized
        structured EDD dataset.

        :param filename: Name of the input .txt file provided to SPEC
            for EDD dataset collection.
        :type filename: str
        :param dataset_id: Number of the dataset in the .txt file to
            return
            `CHAP.common.SetupNXdataProcessor.process`
            arguments for.
        :type dataset_id: int
        :returns: The dataset's coordinate names, values, attributes,
            and signal names, shapes, and attributes.
        :rtype: dict
        """
        # Columns in input .txt file:
        # 0: scan number
        # 1: dataset index
        # 2: configuration descriptor
        # 3: labx
        # 4: laby
        # 5: labz
        # 6: omega (reference)
        # 7: omega (offset)
        # 8: dwell time
        # 9: beam width
        # 10: beam height
        # 11: detector slit gap width
        # 12: scan type

        # Following columns used only for scan types 1 and up and
        # specify flyscan/flymesh parameters.
        # 13 + 4n: scan direction axis index
        # 14 + 4n: lower bound
        # 15 + 4n: upper bound
        # 16 + 4n: no. points
        # (For scan types 1, 4: n = 1)
        # (For scan types 2, 3, 5: n = 1 or 2)

        # For scan type 5 only:
        # 21: bin axis

        import numpy as np

        # Parse dataset from the input .txt file.
        with open(filename) as inf:
            file_lines = inf.readlines()
        dataset_lines = []
        for l in file_lines:
            vals = l.split()
            for i, v in enumerate(vals):
                try:
                    vals[i] = int(v)
                except:
                    try:
                        vals[i] = float(v)
                    except:
                        pass
            if vals[1] == dataset_id:
                dataset_lines.append(vals)

        # Start inferring coords and signals lists for EDD experiments
        self.logger.warning(
            'Assuming the following parameters are identical across the '
            + 'entire dataset: scan type, configuration descriptor')
        scan_type = dataset_lines[0][12]
        self.logger.debug(f'scan_type = {scan_type}')
        coords = [
            dict(name='labx',
                 values=np.unique([l[3] for l in dataset_lines]),
                 attrs=dict(
                     units='mm', local_name='labx', data_type='smb_par')),
            dict(name='laby',
                 values=np.unique([l[4] for l in dataset_lines]),
                 attrs=dict(
                     units='mm', local_name='laby', data_type='smb_par')),
            dict(name='labz',
                 values=np.unique([l[5] for l in dataset_lines]),
                 attrs=dict(
                     units='mm', local_name='labz', data_type='smb_par')),
            dict(name='ometotal',
                 values=np.unique([l[6] + l[7] for l in dataset_lines]),
                 attrs=dict(
                     units='degrees', local_name='ometotal',
                     data_type='smb_par'))
        ]
        signals = [
            dict(name='presample_intensity', shape=[],
                 attrs=dict(units='counts', local_name='a3ic1',
                            data_type='scan_column')),
            dict(name='postsample_intensity', shape=[],
                 attrs=dict(units='counts', local_name='diode',
                            data_type='scan_column')),
            dict(name='dwell_time_actual', shape=[],
                 attrs=dict(units='seconds', local_name='sec',
                            data_type='scan_column')),
            dict(name='SCAN_N', shape=[],
                 attrs=dict(units='n/a', local_name='SCAN_N',
                            data_type='smb_par')),
            dict(name='rsgap_size', shape=[],
                 attrs=dict(units='mm', local_name='rsgap_size',
                            data_type='smb_par')),
            dict(name='x_effective', shape=[],
                 attrs=dict(units='mm', local_name='x_effective',
                            data_type='smb_par')),
            dict(name='z_effective', shape=[],
                 attrs=dict(units='mm', local_name='z_effective',
                            data_type='smb_par')),
        ]
        for i in range(23):
            signals.append(dict(
                name=str(i), shape=[4096,],
                attrs=dict(
                    units='counts', local_name=f'XPS23 element {i}',
                    eta='unknown')))

        attrs = dict(dataset_id=dataset_id,
                     config_id=dataset_lines[0][2],
                     scan_type=scan_type)

        # For potential coordinate axes w/ only one unique value, do
        # not consider them a coordinate. Make them a signal instead.
        _coords = []
        for i, c in enumerate(coords):
            if len(c['values']) == 1:
                self.logger.debug(f'Moving {c["name"]} from coords to signals')
                # signal = coords.pop(i)
                del c['values']
                c['shape'] = []
                signals.append(c)
            else:
                _coords.append(c)
        coords = _coords

        # Append additional coords depending on the scan type of the
        # dataset. Also find the number of points / scan.
        if scan_type == 0:
            scan_npts = 1
        else:
            self.logger.warning(
                'Assuming scan parameters are identical for all scans.')
            axes_labels = {1: 'scan_labx', 2: 'scan_laby', 3: 'scan_labz',
                           4: 'scan_ometotal'}
            axes_units = {1: 'mm', 2: 'mm', 3: 'mm', 4: 'degrees'}
            coords.append(
                dict(name=axes_labels[dataset_lines[0][13]],
                     values=np.round(np.linspace(
                         dataset_lines[0][14], dataset_lines[0][15],
                         dataset_lines[0][16]), 3),
                     attrs=dict(units=axes_units[dataset_lines[0][13]],
                                relative=True)))
            scan_npts = len(coords[-1]['values'])
            if scan_type in (2, 3, 5):
                coords.append(
                    dict(name=axes_labels[dataset_lines[0][17]],
                         values=np.round(np.linspace(
                             dataset_lines[0][18], dataset_lines[0][19],
                             dataset_lines[0][20]), 3),
                         attrs=dict(units=axes_units[dataset_lines[0][17]],
                                    relative=True)))
                scan_npts *= len(coords[-1]['values'])
                if scan_type == 5:
                    attrs['bin_axis'] = axes_labels[dataset_lines[0][21]]

        # Determine if the datset is structured or unstructured.
        total_npts = len(dataset_lines) * scan_npts
        self.logger.debug(f'Total # of points in the dataset: {total_npts}')
        self.logger.debug(
            'Determined number of unique coordinate values: '
            + str({c['name']: len(c['values']) for c in coords}))
        coords_npts = np.prod([len(c['values']) for c in coords])
        self.logger.debug(
            f'If dataset is structured, # of points should be: {coords_npts}')
        if coords_npts != total_npts:
            attrs['unstructured_axes'] = []
            self.logger.warning(
                'Dataset is unstructured. All coordinates will be treated as '
                + 'singals, and the dataset will have a single coordinate '
                + 'instead: data point index.')
            for c in coords:
                del c['values']
                c['shape'] = []
                signals.append(c)
                attrs['unstructured_axes'].append(c['name'])
            coords = [dict(name='dataset_point_index',
                           values=np.arange(total_npts),
                           attrs=dict(units='n/a'))]
        else:
            signals.append(dict(name='dataset_point_index', shape=[],
                                attrs=dict(units='n/a')))

        return dict(coords=coords, signals=signals, attrs=attrs)


class UpdateNXdataReader(Reader):
    """Companion to `edd.SetupNXdataReader` and
    `common.UpdateNXDataProcessor`. Constructs a list of data points
    to pass as pipeline data to `common.UpdateNXDataProcessor` so that
    an `NXdata` constructed by `edd.SetupNXdataReader` and
    `common.SetupNXdataProcessor` can be updated live as individual
    scans in an EDD dataset are completed.

    Example of use in a `Pipeline` configuration:
    ```yaml
    config:
      inputdir: /rawdata/samplename
    pipeline:
      - edd.UpdateNXdataReader:
          spec_file: spec.log
          scan_number: 1
      - common.SetupNXdataProcessor:
          nxfilename: /reduceddata/samplename/data.nxs
          nxdata_path: /entry/samplename_dataset_1
    ```
    """
    def read(self, spec_file, scan_number, inputdir='.'):
        """Return a list of data points containing raw data values for
        a single EDD spec scan. The returned values can be passed
        along to `common.UpdateNXdataProcessor` to fill in an existing
        `NXdata` set up with `common.SetupNXdataProcessor`.

        :param spec_file: Name of the spec file containing the spec
           scan (a relative or absolute path).
        :type spec_file: str
        :param scan_number: Number of the spec scan.
        :type scan_number: int
        :param inputdir: Parent directory of `spec_file`, used only if
            `spec_file` is a relative path. Will be ignored if
            `spec_file` is an absolute path. Defaults to `'.'`.
        :type inputdir: str
        :returs: List of data points appropriate for input to
            `common.UpdateNXdataProcessor`.
        :rtype: list[dict[str, object]]
        """
        import os
        from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser
        from CHAP.utils.parfile import ParFile

        if not os.path.isabs(spec_file):
            spec_file = os.path.join(inputdir, spec_file)
        scanparser = ScanParser(spec_file, scan_number)
        self.logger.debug('Parsed scan')

        # A label / counter mne dict for convenience
        counters = dict(presample_intensity='a3ic0',
                        postsample_intensity='diode',
                        dwell_time_actual='sec')
        # Determine the scan's own coordinate axes based on scan type
        scan_type = scanparser.pars['scan_type']
        self.logger.debug(f'scan_type = {scan_type}')
        if scan_type == 0:
            scan_axes = []
        else:
            axes_labels = {1: 'scan_labx', 2: 'scan_laby', 3: 'scan_labz',
                           4: 'scan_ometotal'}
            scan_axes = [axes_labels[scanparser.pars['fly_axis0']]]
            if scan_type in (2, 3, 5):
                scan_axes.append(axes_labels[scanparser.pars['fly_axis1']])
        self.logger.debug(f'Determined scan axes: {scan_axes}')

        # Par file values will be the same for all points in any scan
        smb_par_values = {}
        for smb_par in ('labx', 'laby', 'labz', 'ometotal', 'SCAN_N',
                        'rsgap_size', 'x_effective', 'z_effective'):
            smb_par_values[smb_par] = scanparser.pars[smb_par]

        # Get offset for the starting index of this scan's points in
        # the entire dataset.
        dataset_id = scanparser.pars['dataset_id']
        parfile = ParFile(scanparser._par_file)
        good_scans = parfile.good_scan_numbers()
        n_prior_dataset_scans = sum(
            [1 if did == dataset_id and scan_n < scan_number else 0 \
             for did, scan_n in zip(
                     parfile.get_values(
                         'dataset_id', scan_numbers=good_scans),
                     good_scans)])
        dataset_point_index_offset = n_prior_dataset_scans \
                                     * scanparser.spec_scan_npts
        self.logger.debug(
            f'dataset_point_index_offset = {dataset_point_index_offset}')

        # Get full data point for every point in the scan
        data_points = []
        self.logger.info(f'Getting {scanparser.spec_scan_npts} data points')
        for i in range(scanparser.spec_scan_npts):
            self.logger.debug(f'Getting data point for scan step index {i}')
            step = scanparser.get_scan_step(i)
            data_points.append(dict(
                dataset_point_index=dataset_point_index_offset + i,
                **smb_par_values,
                **{str(_i): scanparser.get_detector_data(_i, i) \
                   for _i in range(23)},
                **{c: scanparser.spec_scan_data[counters[c]][i] \
                   for c in counters},
                **{a: round(
                    scanparser.spec_scan_motor_vals_relative[_i][step[_i]], 3)\
                   for _i, a in enumerate(scan_axes)}))

        return data_points


class NXdataSliceReader(Reader):
    """Reader for returning a sliced verison of an `NXdata` (which
    represents a full EDD dataset) that contains data from just a
    single SPEC scan.

    Example of use in a `Pipeline` configuration:
    ```yaml
    config:
      inputdir: /rawdata/samplename
      outputdir: /reduceddata/samplename
    pipeline:
      - edd.NXdataSliceReader:
          filename: /reduceddata/samplename/data.nxs
          nxpath: /path/to/nxdata
          spec_file: spec.log
          scan_number: 1
      - common.NexusWriter:
          filename: scan_1.nxs
    ```
    """
    def read(self, filename, nxpath, spec_file, scan_number, inputdir='.'):
        """Return a "slice" of an EDD dataset's NXdata that represents
        just the data from one scan in the dataset.

        :param filename: Name of the NeXus file in which the
            existing full EDD dataset's NXdata resides.
        :type filename: str
        :param nxpath: Path to the existing full EDD dataset's NXdata
            group in `filename`.
        :type nxpath: str
        :param spec_file: Name of the spec file containing whose data
            will be the only contents of the returned `NXdata`.
        :type spec_file: str
        :param scan_number: Number of the spec scan whose data will be
            the only contents of the returned `NXdata`.
        :type scan_number: int
        :param inputdir: Directory containing `filename` and/or
            `spec_file`, if either one / both of them are not absolute
            paths. Defaults to `'.'`.
        :type inputdir: str, optional
        :returns: An `NXdata` similar to the one at `nxpath` in
            `filename`, but containing only the data collected by the
            specified spec scan.
        :rtype: nexusformat.nexus.NXdata
        """
        import os

        from nexusformat.nexus import nxload
        import numpy as np

        from CHAP.common import NXdataReader
        from CHAP.utils.parfile import ParFile
        from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser

        # Parse existing NXdata
        root = nxload(filename)
        nxdata = root[nxpath]
        if nxdata.nxclass != 'NXdata':
            raise TypeError(
                f'Object at {nxpath} in {filename} is not an NXdata')
        self.logger.debug('Loaded existing NXdata')

        # Parse scan
        if not os.path.isabs(spec_file):
            spec_file = os.path.join(inputdir, spec_file)
        scanparser = ScanParser(spec_file, scan_number)
        self.logger.debug('Parsed scan')

        # Assemble arguments for NXdataReader
        name = f'{nxdata.nxname}_scan_{scan_number}'
        axes_names = [a.nxname for a in nxdata.nxaxes]
        if nxdata.nxsignal is not None:
            signal_name = nxdata.nxsignal.nxname
        else:
            signal_name = list(nxdata.entries.keys())[0]
        attrs = nxdata.attrs
        nxfield_params = []
        if 'dataset_point_index' in nxdata:
            # Get offset for the starting index of this scan's points in
            # the entire dataset.
            dataset_id = scanparser.pars['dataset_id']
            parfile = ParFile(scanparser._par_file)
            good_scans = parfile.good_scan_numbers()
            n_prior_dataset_scans = sum(
                [1 if did == dataset_id and scan_n < scan_number else 0 \
                 for did, scan_n in zip(
                         parfile.get_values(
                             'dataset_id', scan_numbers=good_scans),
                         good_scans)])
            dataset_point_index_offset = n_prior_dataset_scans \
                                         * scanparser.spec_scan_npts
            self.logger.debug(
                f'dataset_point_index_offset = {dataset_point_index_offset}')
            slice_params = dict(
                start=dataset_point_index_offset,
                end=dataset_point_index_offset + scanparser.spec_scan_npts + 1)
            nxfield_params = [dict(filename=filename,
                                   nxpath=entry.nxpath,
                                   slice_params=[slice_params]) \
                              for entry in nxdata]
        else:
            signal_slice_params = []
            for a in nxdata.nxaxes:
                if a.nxname.startswith('scan_'):
                    slice_params = {}
                else:
                    value = scanparser.pars[a.nxname]
                    try:
                        index = np.where(a.nxdata == value)[0][0]
                    except:
                        index = np.argmin(np.abs(a.nxdata - value))
                        self.logger.warning(
                            f'Nearest match for coordinate value {a.nxname}: '
                            f'{a.nxdata[index]} (actual value: {value})')
                    slice_params = dict(start=index, end=index+1)
                signal_slice_params.append(slice_params)
                nxfield_params.append(dict(
                    filename=filename,
                    nxpath=os.path.join(nxdata.nxpath, a.nxname),
                    slice_params=[slice_params]))
            for name, entry in nxdata.entries.items():
                if entry in nxdata.nxaxes:
                    continue
                nxfield_params.append(dict(
                    filename=filename, nxpath=entry.nxpath,
                    slice_params=signal_slice_params))

        # Return the "sliced" NXdata
        reader = NXdataReader()
        reader.logger = self.logger
        return reader.read(name=nxdata.nxname, nxfield_params=nxfield_params,
                           signal_name=signal_name, axes_names=axes_names,
                           attrs=attrs)


if __name__ == '__main__':
    from CHAP.reader import main
    main()
