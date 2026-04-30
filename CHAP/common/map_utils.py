"""Common map data model functions and classes."""

from pydantic import (
    conint,
    conlist,
    model_validator,
    Field,
    FilePath,
)
from typing import Optional

# Local modules
from CHAP import Processor
from CHAP.common.models.map import (
    Detector,
    MapConfig,
)

def get_axes(nxdata, skip_axes=None):
    """Get the axes of an NXdata object."""
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

    :ivar map_config: Map configuration.
    :type map_config: CHAP.common.models.map.MapConfig
    :ivar detectors: List os detector configurations.
    :type detectors: list[CHAP.common.models.map.Detector]
    :ivar spec_file: SPEC file containing scan from which to read a
        slice of raw data.
    :type spec_file: pydantic.FilePath
    :ivar scan_numbers: Numbers of scans from which to read slices of
        raw data.
    :type scan_numbers: list[int]
    """
    pipeline_fields: dict = Field(
        default={
            'map_config': 'common.models.map.MapConfig',
        },
        init_var=True)
    map_config: MapConfig
    detectors: conlist(item_type=Detector, min_length=1)
    spec_file: FilePath
    scan_number: Optional[conint(gt=0)] = None
    scan_numbers: Optional[conlist(item_type=conint(gt=0))] = None

    def process(self, data,
                idx_slice={'start': 0, 'step': 1}):
        """Aggregate partial spec and detector data from one or more
        scans in a map, returning results in a format suitable for
        writing to the full map container with
        `common.NexusValuesWriter` or `common.ZarrValuesWriter`.

        When all scans are adjacent in the map and `idx_slice` covers
        each scan in full, data_points entries for the same path are
        consolidated into a single array + slice, avoiding redundant
        write calls.

        :param data: Result of `Reader.read` where at least one item
            has the value `'common.models.map.MapConfig'` for the
            `'schema'` key.
        :type data: list[PipelineData]
        :param idx_slice: Parameters for the slice of each scan to
            process (slice parameters are the usual for the python
            `slice` object: `'start'`, `'stop'`, and `'step'`).
            Defaults to `{'start': 0, 'step': 1}`.
        :type idx_slice: dict[str, int], optional
        :return: Slice of map data, ready to be written to a map
             container.
        :rtype: list[dict[str, object]]
        """
        import numpy as np
        import os
        from CHAP.common.models.map import SpecScans

        scans_obj = SpecScans(
            spec_file=self.spec_file, scan_numbers=self.scan_numbers)
        self_spec_file_abs = os.path.abspath(str(self.spec_file))

        if self.map_config.experiment_type == 'EDD':
            def get_detector_data(scan, detector, index):
                return scan.get_detector_data(detector.get_id(), index)[0][0]
        else:
            def get_detector_data(scan, detector, index):
                return scan.get_detector_data(detector.get_id(), index)

        # Build flat ordered list of (abs_spec_file, scan_number) for
        # all scans in the map to determine each scan's map position
        map_scan_order = []
        for spec_scans_item in self.map_config.spec_scans:
            sf_abs = os.path.abspath(str(spec_scans_item.spec_file))
            for sn in spec_scans_item.scan_numbers:
                map_scan_order.append((sf_abs, sn))
        scan_positions = {}
        for sn in self.scan_numbers:
            for pos, (sf, n) in enumerate(map_scan_order):
                if sf == self_spec_file_abs and n == sn:
                    scan_positions[sn] = pos
                    break

        # Process scans in map order
        sorted_scan_numbers = sorted(
            self.scan_numbers, key=lambda sn: scan_positions[sn])

        slice_start = idx_slice.get('start', 0)
        slice_step = idx_slice.get('step', 1)

        # Collect per-scan metadata; assumes uniform npts across scans
        # for index_offset calculation (index_offset = map_pos * npts)
        per_scan = []
        for sn in sorted_scan_numbers:
            scan = scans_obj.get_scanparser(sn)
            npts_scan = int(scan.spec_scan_npts)
            index_offset = scan_positions[sn] * npts_scan
            # Cap stop at npts_scan so map_indices and data stay in sync
            slice_stop = min(idx_slice.get('stop', npts_scan), npts_scan)
            scan_indices = range(npts_scan)[
                slice(slice_start, slice_stop, slice_step)]
            map_indices = slice(
                slice_start + index_offset,
                slice_stop + index_offset,
                slice_step,
            )
            per_scan.append({
                'sn': sn, 'scan': scan, 'npts_scan': npts_scan,
                'index_offset': index_offset,
                'scan_indices': scan_indices,
                'map_indices': map_indices,
                'full': (slice_start == 0
                         and slice_step == 1
                         and slice_stop == npts_scan),
            })

        # Consolidate into single data_points entries when all scans
        # are adjacent in the map and idx_slice covers each scan fully
        sorted_positions = [scan_positions[sn] for sn in sorted_scan_numbers]
        scans_are_adjacent = all(
            sorted_positions[i + 1] == sorted_positions[i] + 1
            for i in range(len(sorted_positions) - 1)
        )
        can_consolidate = (
            len(per_scan) > 1
            and scans_are_adjacent
            and all(ps['full'] for ps in per_scan)
        )

        if can_consolidate:
            first, last = per_scan[0], per_scan[-1]
            merged_idx = slice(
                first['index_offset'],
                last['index_offset'] + last['npts_scan'],
                1,
            )
            data_points = [{
                'path': (f'{self.map_config.title}'
                         f'/independent_dimensions/index'),
                'data': np.arange(
                    first['index_offset'],
                    last['index_offset'] + last['npts_scan'],
                ),
                'idx': merged_idx,
            }]
            for s_d in self.map_config.all_scalar_data:
                data_points.append({
                    'path': (f'{self.map_config.title}'
                             f'/scalar_data/{s_d.label}'),
                    'data': np.concatenate([
                        np.asarray([
                            s_d.get_value(
                                scans_obj, ps['sn'], i,
                                scalar_data=self.map_config.scalar_data)
                            for i in ps['scan_indices']
                        ])
                        for ps in per_scan
                    ]),
                    'idx': merged_idx,
                })
            for dim in self.map_config.independent_dimensions:
                data_points.append({
                    'path': (f'{self.map_config.title}'
                             f'/independent_dimensions/{dim.label}'),
                    'data': np.concatenate([
                        np.asarray([
                            dim.get_value(
                                scans_obj, ps['sn'], i,
                                scalar_data=self.map_config.scalar_data)
                            for i in ps['scan_indices']
                        ])
                        for ps in per_scan
                    ]),
                    'idx': merged_idx,
                })
            for det in self.detectors:
                data_points.append({
                    'path': (f'{self.map_config.title}'
                             f'/data/{det.get_id()}'),
                    'data': np.concatenate([
                        np.asarray([
                            get_detector_data(ps['scan'], det, i)
                            for i in ps['scan_indices']
                        ])
                        for ps in per_scan
                    ]),
                    'idx': merged_idx,
                })
        else:
            data_points = []
            for ps in per_scan:
                data_points.append({
                    'path': (f'{self.map_config.title}'
                             f'/independent_dimensions/index'),
                    'data': np.asarray(
                        [ps['index_offset'] + i for i in ps['scan_indices']]
                    ),
                    'idx': ps['map_indices'],
                })
                data_points.extend([{
                    'path': (f'{self.map_config.title}'
                             f'/scalar_data/{s_d.label}'),
                    'data': np.asarray([
                        s_d.get_value(
                            scans_obj, ps['sn'], i,
                            scalar_data=self.map_config.scalar_data)
                        for i in ps['scan_indices']
                    ]),
                    'idx': ps['map_indices'],
                } for s_d in self.map_config.all_scalar_data])
                data_points.extend([{
                    'path': (f'{self.map_config.title}'
                             f'/independent_dimensions/{dim.label}'),
                    'data': np.asarray([
                        dim.get_value(
                            scans_obj, ps['sn'], i,
                            scalar_data=self.map_config.scalar_data)
                        for i in ps['scan_indices']
                    ]),
                    'idx': ps['map_indices'],
                } for dim in self.map_config.independent_dimensions])
                data_points.extend([{
                    'path': (f'{self.map_config.title}'
                             f'/data/{det.get_id()}'),
                    'data': np.asarray([
                        get_detector_data(ps['scan'], det, i)
                        for i in ps['scan_indices']
                    ]),
                    'idx': ps['map_indices'],
                } for det in self.detectors])
        return data_points

    @model_validator(mode='before')
    @classmethod
    def fill_scan_numbers(cls, data):
        if not isinstance(data, dict):
            return data
        if 'scan_numbers' not in data or data['scan_numbers'] is None:
            if data.get('scan_number') is not None:
                data['scan_numbers'] = [data['scan_number']]
        elif isinstance(data['scan_numbers'], int):
            data['scan_numbers'] = [data['scan_numbers']]
        elif isinstance(data['scan_numbers'], str):
            from CHAP.utils.general import string_to_list
            data['scan_numbers'] = string_to_list(data['scan_numbers'])
        return data

    @model_validator(mode='after')
    def validate_scan_numbers(self):
        if self.scan_numbers is None:
            raise ValueError(
                'scan_numbers is required; alternatively, provide scan_number')
        if self.scan_number is not None \
           and self.scan_number not in self.scan_numbers:
            self.scan_numbers.append(self.scan_number)
        return self


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
