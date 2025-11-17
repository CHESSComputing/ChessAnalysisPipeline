"""Common map data model functions and classes."""

# Local modules
from CHAP import Processor

def get_axes(nxdata, skip_axes=None):
    """Get the axes of an NXdata object used in EDD."""
    if skip_axes is None:
        skip_axes = []
    if 'unstructured_axes' in nxdata.attrs:
        axes = nxdata.attrs['unstructured_axes']
    elif 'axes' in nxdata.attrs:
        axes = nxdata.attrs['axes']
    else:
        return []
    if isinstance(axes, str):
        axes = [axes]
    return [str(a) for a in axes if a not in skip_axes]


class MapSliceProcessor(Processor):
    """Proccessor for getting partial map data for filling in a NeXus
    structure created by `CHAP.common.MapProcessor` with
    `fill_data=False`. Good for parallelizing workflows across
    multiple pipelines or processing data live when a scan is still
    incomplete. Returned data is suitable for writing to an existing
    map structure with `common.NexusValuesWriter` or
    `common.ZarrValuesWriter`.
    """
    def process(self, data, spec_file, scan_number,
                idx_slice={'start': 0, 'step': 1},
                detectors=None,
                config=None,
                inputdir='.'):
        """Aggregate partial spec and detector data from one scan in a
        map, returning results in a format suitable for writing to the
        full map container with `common.NexusValuesWriter` or
        `common.ZarrValuesWriter`.

        :param data: Result of `Reader.read` where at least one item
            has the value `'common.models.map.MapConfig'` for the
            `'schema'` key.
        :type data: list[PipelineData]
        :param spec_file: Name of spec file containing scan data to be
            processed.
        :type spec_file: str
        :param scan_number: Number of scan containing data to be processed.
        :type scan_number: int
        :type idx_slice: Parameters for the slice of the scan to
            process (slice parameters are the usual for the python
            `slice` object: `'start'`, `'stop'`, and
            `'step'`). Defaults to `{'start': 0, 'step': '1'}`.
        :type idx_slice: dict[str, int], optional
        :param detectors: Detectors to include raw data for in the
            returned NeXus NXentry object (overruling the detector
            info in data, if present).
        :type detectors: list[dict], optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths.
        :type inputdir: str, optional
        :return: Slice of map data, ready to be written to a map
             container.
        :rtype: list[dict[str, object]]
        """
        import numpy as np
        import os
        from chess_scanparsers import choose_scanparser

        from CHAP.common.models.map import SpecScans

        # Get the validated map configuration
        map_config = self.get_config(
            data=data, config=config, schema='common.models.map.MapConfig',
            inputdir=inputdir)

        # Validate the detectors
        try:
            from CHAP.common.models.map import DetectorConfig
            detector_config = DetectorConfig(detectors=detectors)
        except:
            try:
                detector_config = self.get_config(
                    data=data, schema='common.models.map.DetectorConfig',
                    inputdir=inputdir)
            except Exception as exc:
                raise RuntimeError from exc

        if not os.path.isabs(spec_file):
            spec_file = os.path.join(inputdir, spec_file)

        ScanParser = choose_scanparser(
            map_config.station, map_config.experiment_type)
        scans = SpecScans(spec_file=spec_file, scan_numbers=[scan_number])
        scan = scans.get_scanparser(scan_number)

        # Get index offset for this data slice within the map
        npts_scan = int(scan.spec_scan_npts)
        nscans_prev = 0
        for scans in map_config.spec_scans:
            for scan_n in scans.scan_numbers:
                if scans.spec_file == spec_file and scan_n == scan_number:
                    break
                nscans_prev += 1
        index_offset = nscans_prev * npts_scan

        # Get spec scan indices to process
        scan_indices = range(npts_scan)[slice(
            idx_slice.get('start', 0),
            idx_slice.get('stop', npts_scan + 1),
            idx_slice.get('step', 1)
        )]
        # Get map indices to write to
        map_indices = slice(
            idx_slice.get('start', 0) + index_offset,
            idx_slice.get('stop', npts_scan + 1) + index_offset,
            idx_slice.get('step', 1)
        )

        data_points = [
            {
                'path': f'{map_config.title}/scalar_data/{s_d.label}',
                'data': np.asarray([
                    s_d.get_value(
                        scans, scan_number, i,
                        scalar_data=map_config.scalar_data)
                    for i in scan_indices
                ]),
                'idx': map_indices
            }
            for s_d in map_config.all_scalar_data
        ]
        data_points.extend(
            [
                {
                    'path': f'{map_config.title}/data/{det.id}',
                    'data': np.asarray([
                        scan.get_detector_data(det.id, i)
                        for i in scan_indices
                    ]),
                    'idx': map_indices
                }
                for det in detector_config.detectors
            ]
        )
        return data_points


class SpecScanToMapConfigProcessor(Processor):
    """Processor to get the `CHAP.common.models.map.MapConfig`
    dictionary configuration representation of a single CHESS SPEC
    scan."""
    def process(self, data,
                spec_file, scan_number, station, experiment,
                dwell_time_actual_counter_name,
                presample_intensity_counter_name,
                postsample_intensity_counter_name=None,
                validate_data_present=True):
        """Return a dictionary representing a valid MapConfig object that
        contains only the single given scan.

        :param spec_file: Name of spec file
        :type spec_file: str
        :param scan_number: Number of scan
        :type scan_number: int
        :param station: Station id ("id**" format)
        :type station: Literal["id1a3", "id3a", "id3b", "id4b"]
        :param experiment: Experiment type
        :type experiment: Literal["edd", "giwaxs", "hdrm", "powder",
            "saxswaxs", "tomo", "xrf"]
        :returns: Single-scan map configuration
        :rtype: dict
        """
        import os

        from chess_scanparsers import choose_scanparser

        SP = choose_scanparser(station, experiment)
        sp = SP(spec_file, scan_number)

        def get_independent_dimensions(_scanparser):
            """Return a value for the `independent_dimensions` field of a
            `MapConfig` object containing just one SPEC scan -- the one
            represented by the given `_scanparser`.

            :param _scanparser: The instance of `ScanParser` to get
                `independent_dimensions` for.
            :type _scanparser: chess_scanparsers.FMBSAXSWAXSScanParser
            :returns: Value to use for the `independent_dimensions` field in
                the `MapConfig` associated with this scan.
            :rtype: list[dict[str, str]]
            """
            from datetime import datetime
            import re
            match = re.match(r'a(\d+)scan', _scanparser.spec_macro)
            if match:
                # Use only the first motor as the independent dim. All
                # others, even though they are also scanned, are scalar
                # data
                return (
                    [{'label': _scanparser.spec_scan_motor_mnes[0],
                      'units': 'unknown',
                      'data_type': 'spec_motor',
                      'name': _scanparser.spec_scan_motor_mnes[0]}],
                    [{'label': mne,
                      'units': 'unknown units',
                      'data_type': 'spec_motor',
                      'name': mne}
                     for mne in _scanparser.spec_scan_motor_mnes[1:]]
                )
            if _scanparser.spec_macro in ('tseries', 'loopscan'):
                scan_firstline = _scanparser.spec_scan.firstline
                headers = _scanparser.spec_file._headers
                useheader_i = -1
                while useheader_i < len(headers) - 1:
                    if headers[useheader_i + 1].firstline < scan_firstline:
                        useheader_i += 1
                    else:
                        break
                    t0 = headers[useheader_i]._epoch
                return (
                    [{'label': 'Epoch',
                      'units': 'seconds',
                      'data_type': 'expression',
                      'name': f'Epoch_offset + {t0}'}],
                    [{'label': 'Epoch_offset',
                      'units': 'seconds',
                      'data_type': 'scan_column',
                      'name': 'Epoch'}])
            elif _scanparser.spec_macro == 'flyscan' and \
                 not len(_scanparser.spec_args) == 5:
                return (
                    [{'label': 'Time',
                      'units': 'seconds',
                      'data_type': 'scan_column',
                      'name': 'Time'}],
                    [])
            elif _scanparser.is_snake():
                return (
                    [{'label': mne,
                      'units': 'unknown units',
                      'data_type': 'scan_column',
                      'name': list(_scanparser.spec_scan_data.keys())[i]}
                     for i, mne in enumerate(
                             _scanparser.spec_scan_motor_mnes)],
                    []
                )
            return (
                [{'label': mne,
                  'units': 'unknown units',
                  'data_type': 'spec_motor',
                  'name': mne}
                 for mne in _scanparser.spec_scan_motor_mnes],
                [])

        normalized_spec_file = os.path.realpath(spec_file).replace(
            '/daq/', '/raw/')
        independent_dimensions, scalar_data = get_independent_dimensions(sp)
        mapconfig_dict = {
            'validate_data_present': validate_data_present,
            'title': sp.scan_title,
            'station': station,
            'experiment_type': experiment.upper(),
            'sample': {
                'name': sp.scan_name
            },
            'spec_scans': [
                {
                    'spec_file': normalized_spec_file,
                    'scan_numbers': [scan_number]
                }
            ],
            'independent_dimensions': independent_dimensions,
            'dwell_time_actual': {
                'data_type': 'scan_column',
                'name': dwell_time_actual_counter_name},
            'presample_intensity': {
                'data_type': 'scan_column',
                'name': presample_intensity_counter_name},
            'scalar_data': scalar_data,
        }
        if postsample_intensity_counter_name:
                mapconfig_dict['postsample_intensity'] = {
                    'data_type': 'scan_column',
                    'name': postsample_intensity_counter_name}
        return mapconfig_dict
