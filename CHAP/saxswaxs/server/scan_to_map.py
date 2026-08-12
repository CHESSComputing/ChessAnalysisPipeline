#!/nfs/chess/sw/miniforge3_chap/envs/CHAP_saxswaxs/bin/python
"""Script to convert a single SPEC scan into its
MapConfigRepresentation using CHAP tools
"""

def spec_scan_to_map_config(
        spec_file, scan_number, station, experiment,
        dwell_time_actual_counter_name,
        presample_intensity_counter_name,
        postsample_intensity_counter_name):
    """Convert a single SPEC scan into a CHAP map config data structure.

    :param spec_file: Path to the SPEC file containing the scan.
    :type spec_file: str
    :param scan_number: Number of the scan within the SPEC file.
    :type scan_number: int
    :param station: Station identifier (e.g. ``'id3b'``).
    :type station: str
    :param experiment: Experiment type (e.g. ``'SAXSWAXS'``).
    :type experiment: str
    :param dwell_time_actual_counter_name: SPEC counter column name for actual
        dwell times.
    :type dwell_time_actual_counter_name: str
    :param presample_intensity_counter_name: SPEC counter column name for
        presample intensity.
    :type presample_intensity_counter_name: str
    :param postsample_intensity_counter_name: SPEC counter column name for
        postsample intensity, or ``None`` if not recorded.
    :type postsample_intensity_counter_name: str or None
    :returns: Map config data structure.
    :rtype: dict
    """
    from CHAP.common.map_utils import SpecScanToMapConfigProcessor

    proc = SpecScanToMapConfigProcessor()
    map_config = proc.process(
        None,
        spec_file, scan_number, station, experiment,
        dwell_time_actual_counter_name,
        presample_intensity_counter_name,
        postsample_intensity_counter_name,
        validate_data_present=False,
    )
    return map_config

def write_yaml(data, filename):
    """Write data to a YAML file, creating parent directories as needed.

    :param data: Data to serialize to YAML.
    :param filename: Output file path.
    :type filename: str
    """
    import os
    import yaml

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outf:
        yaml.dump(data, outf, sort_keys=False)

def scan_to_map(
        spec_file, scan_number, station, experiment,
        dwell_time_actual_counter_name,
        presample_intensity_counter_name,
        postsample_intensity_counter_name,
        map_config_filename):
    """Convert a SPEC scan to a map config YAML file.

    :param spec_file: Path to the SPEC file containing the scan.
    :type spec_file: str
    :param scan_number: Number of the scan within the SPEC file.
    :type scan_number: int
    :param station: Station identifier (e.g. ``'id3b'``).
    :type station: str
    :param experiment: Experiment type (e.g. ``'SAXSWAXS'``).
    :type experiment: str
    :param dwell_time_actual_counter_name: SPEC counter column name for actual
        dwell times.
    :type dwell_time_actual_counter_name: str
    :param presample_intensity_counter_name: SPEC counter column name for
        presample intensity.
    :type presample_intensity_counter_name: str
    :param postsample_intensity_counter_name: SPEC counter column name for
        postsample intensity, or ``None`` if not recorded.
    :type postsample_intensity_counter_name: str or None
    :param map_config_filename: Output path for the map config YAML file.
    :type map_config_filename: str
    """
    map_config_data = spec_scan_to_map_config(
        spec_file, scan_number, station, experiment,
        dwell_time_actual_counter_name,
        presample_intensity_counter_name,
        postsample_intensity_counter_name,
    )
    write_yaml(map_config_data, map_config_filename)


if __name__ == '__main__':
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description='''Create a map_config.yaml representing the given
        SPEC scan''')
    parser.add_argument(
        '--spec_file', required=True,
        help='Name of the SPEC file containing the scan of interest.')
    parser.add_argument(
        '--scan_number', required=True, type=int,
        help='Number of the scan of interest within the given SPEC file.')
    parser.add_argument(
        '--outputdir', required=True, help='''Path to the output
        analysis directory for this scan. The map configuration YAML
        and reduced data NeXus files will be written here.''')
    parser.add_argument(
        '--dwell_time_actual_counter_name', required=True,
        help='''Name of the SPEC counter column representing actual
        dwell times''')
    parser.add_argument(
        '--presample_intensity_counter_name', required=True,
        help='''Name of the SPEC counter column representing presample
        intensity values''')
    parser.add_argument(
        '--postsample_intensity_counter_name',
        help='''Name of the SPEC counter column representing postsample
        intensity values''')
    parser.add_argument(
        '--station', choices=['id3b'], default='id3b',
        help='''Name of the station at which the scan was collected
        ("id3b" is currently the only supported value)''')
    parser.add_argument(
        '--experiment', choices=['SAXSWAXS'], default='SAXSWAXS',
        help='''Name of the scan\'s experiment type ("SAXSWAXS" is
        currently the only supported value).''')
    args = parser.parse_args(sys.argv[1:])

    scan_to_map(
        args.spec_file, args.scan_number, args.station, args.experiment,
        args.dwell_time_actual_counter_name,
        args.presample_intensity_counter_name,
        args.postsample_intensity_counter_name,
        os.path.join(args.outputdir, 'map_config.yaml')
    )
