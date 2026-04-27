#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Common map data model functions and classes."""

# Third party modules
from pydantic import (
    conint,
    conlist,
    Field,
    FilePath,
)

# Local modules
from CHAP.processor import Processor
from CHAP.common.models.map import (
    Detector,
    MapConfig,
)

def get_axes(nxdata, skip_axes=None):
    """Get the axes of a NeXus style
    `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html#index-0>`__
    object.

    :param nxdata: Input data.
    :type nxdata: nexusformat.nexus.NXdata
    :param skip_axes: Axes to skip.
    :type skip_axes: list[str], optional
    :return: The axes of the NXdata object excluding those in the
        optional `skip_axes` parameter.
    :rtype: list[str]
    """
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
    structure created by :class:`~CHAP.common.processor.MapProcessor`
    with `fill_data=False`. Good for parallelizing workflows across
    multiple pipelines or processing data live when a scan is still
    incomplete. Returned data is suitable for writing to an existing
    map structure with :class:`~CHAP.common.writer.NexusValuesWriter`
    or :class:`~CHAP.common.writer.ZarrValuesWriter`.

    :ivar map_config: Map configuration.
    :vartype map_config: MapConfig
    :ivar detectors: Detector configurations.
    :vartype detectors:
        list[:class:`~CHAP.common.models.map.Detector`]
    :ivar spec_file: SPEC file containing scan from which to read a
        slice of raw data.
    :vartype spec_file: str
    :ivar scan_number: Number of scan from which to read a slice of
        raw data.
    :vartype scan_number: int
    """

    pipeline_fields: dict = Field(
        default={
            'map_config': 'common.models.map.MapConfig',
        },
        init_var=True)
    map_config: MapConfig
    detectors: conlist(item_type=Detector, min_length=1)
    spec_file: FilePath
    scan_number: conint(gt=0)

    def process(self, data, #spec_file, scan_number,
                idx_slice={'start': 0, 'step': 1}):
        """Aggregate partial spec and detector data from one scan in a
        map, returning results in a format suitable for writing to the
        full map container with
        :class:`~CHAP.common.writer.NexusValuesWriter` or
        :class:`~CHAP.common.writer.ZarrValuesWriter`.

        :param data: Result of `Reader.read` where at least one item
            has the value `'common.models.map.MapConfig'` for the
            `'schema'` key.
        :type data: list[PipelineData]
        :type idx_slice: Parameters for the slice of the scan to
            process (slice parameters are the usual for the python
            `slice` object: `'start'`, `'stop'`, and
            `'step'`). Defaults to `{'start': 0, 'step': '1'}`.
        :type idx_slice: dict[str, int], optional
        :return: Slice of map data, ready to be written to a map
             container.
        :rtype: list[dict[str, Any]]
        """
        # System modules
        import os

        # Third party modules
        import numpy as np

        # Local modules
        from chess_scanparsers import choose_scanparser
        from CHAP.common.models.map import SpecScans

        ScanParser = choose_scanparser(
            self.map_config.station, self.map_config.experiment_type)
        scans = SpecScans(
            spec_file=self.spec_file, scan_numbers=[self.scan_number])
        scan = scans.get_scanparser(self.scan_number)

        # Get index offset for this data slice within the map
        npts_scan = int(scan.spec_scan_npts)
        nscans_prev = 0
        for scans in self.map_config.spec_scans:
            for scan_n in scans.scan_numbers:
                if (os.path.abspath(self.spec_file) == \
                    os.path.abspath(self.spec_file)
                    and scan_n == self.scan_number):
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
                'path': f'{self.map_config.title}/scalar_data/{s_d.label}',
                'data': np.asarray([
                    s_d.get_value(
                        scans, self.scan_number, i,
                        scalar_data=self.map_config.scalar_data)
                    for i in scan_indices
                ]),
                'idx': map_indices
            }
            for s_d in self.map_config.all_scalar_data
        ]
        data_points.extend(
            [
                {
                    'path': f'{self.map_config.title}/data/{det.get_id()}',
                    'data': np.asarray([
                        scan.get_detector_data(det.get_id(), i)
                        for i in scan_indices
                    ]),
                    'idx': map_indices
                }
                for det in self.detectors
            ]
        )
        return data_points


class SpecScanToMapConfigProcessor(Processor):
    """Processor to get the
    :class:`~CHAP.common.models.map.MapConfig` dictionary
    configuration representation of a single CHESS SPEC scan.
    """

    def process(self, data,
                spec_file, scan_number, station, experiment,
                dwell_time_actual_counter_name,
                presample_intensity_counter_name,
                postsample_intensity_counter_name=None,
                validate_data_present=True):
        """Return a dictionary representing a valid
        :class:`~CHAP.common.models.map.MapConfig` object that
        contains only the single given scan.

        :param spec_file: Spec file name
        :type spec_file: str
        :param scan_number: Scan number
        :type scan_number: int
        :param station: Name of the station at which the data was
            collected.
        :type station: Literal["id1a3", "id3a", "id3b", "id4b"]
        :param experiment: Experiment type
        :type experiment_type: Literal[
            'EDD', 'GIWAXS', 'HDRM', 'SAXSWAXS', 'TOMO', 'XRF']
        :param dwell_time_actual_counter_name: Name of the counter used
            to record the actual dwell time at time of data collection.
        :type dwell_time_actual_counter_name: str
        :param presample_intensity_counter_name: Name of the counter
            used to record the incident beam intensity at time of data
            collection.
        :type presample_intensity_counter_name: str
        :param postsample_intensity_counter_name: Name of the counter
            used to record the post sample beam intensity at time of
            data collection.
        :type postsample_intensity_counter_name: str, optional
        :param validate_data_present: Optional `validate_data_present`
            key-value pair to the output map configuration, defaults
            to `True`.
        :type validate_data_present:
        :returns: Single-scan map configuration
        :rtype: dict
        """
        # System modules
        import os

        # Local modules
        from chess_scanparsers import choose_scanparser

        SP = choose_scanparser(station, experiment)
        sp = SP(spec_file, scan_number)

        def get_independent_dimensions(_scanparser):
            """Return a value for the `independent_dimensions` field of
            a :class:`~CHAP.common.models.map.MapConfig` object
            containing just one SPEC scan -- the one represented by the
            given `_scanparser`.

            :param _scanparser: The instance of
                `ScanParser <https://github.com/CHESSComputing/chess-scanparsers?tab=readme-ov-file>`__
                to get `independent_dimensions` for.
            :type _scanparser: chess_scanparsers.ScanParser
            :returns: Value to use for the `independent_dimensions`
                field in the
                :class:`~CHAP.common.models.map.MapConfig` object
                associated with this scan.
            :rtype: list[dict[str, str]]
            """
            # System modules
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
            if _scanparser.spec_macro == 'flyscan' and \
                 not len(_scanparser.spec_args) == 5:
                return (
                    [{'label': 'Time',
                      'units': 'seconds',
                      'data_type': 'scan_column',
                      'name': 'Time'}],
                    [])
            if _scanparser.is_snake():
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
                'name': postsample_intensity_counter_name,
            }
        return mapconfig_dict
