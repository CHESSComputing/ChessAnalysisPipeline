from CHAP import Processor

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
                idx_slice={'start': 0, 'stop': -1, 'step': 1},
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
            `'step'`). Defaults to `{'start': 0, 'stop': -1, 'step':
            '1'}`.
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
        
