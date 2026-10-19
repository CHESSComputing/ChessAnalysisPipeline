"""
Contains a CLI tool to convert raw FMB-style XRF data to a standard
`NXentry` structure with :class:`~CHAP.common.processor.MapProcessor`.
"""
import argparse
import logging
import os
import sys
from typing import Literal

from CHAP.common.map_utils import SpecScanToMapConfigProcessor
from CHAP.common.processor import MapProcessor
from CHAP.common.writer import NexusWriter
from CHAP.pipeline import PipelineData


VORTEX_CONFIG = {
    'detectors': [
        {'id': i}
        for i in range(3)
    ]
}


logger = logging.getLogger('xrf-map')
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter(
    '{asctime}: {name:20} (L{lineno}): {levelname}: {message}',
    datefmt='%Y-%m-%d %H:%M:%S', style='{'))
logger.handlers = [log_handler]

def get_map_config(args):
    """Return a dictionary that represents a valid
    :class"`~CHAP.common.models.map.MapConfig` object for the XRF
    dataset identified by `args`.

    :param args: CLI arguments
    :type args: CLIConfig
    :returns: Map configuration
    :rtype: dict
    """
    proc = SpecScanToMapConfigProcessor(
        spec_file=args.spec_file,
        scan_number=args.scan_number,
        station=args.station,
        experiment=args.experiment,
        dwell_time_actual_counter_name=args.dwell_time_actual_counter_name,
        presample_intensity_counter_name=args.presample_intensity_counter_name,
        postsample_intensity_counter_name=args.postsample_intensity_counter_name,
        validate_data_present=False,
        logger=logger,
    )
    return proc.process([])


def process_map_config(mapprocessor_config):
    """Run :class:`~CHAP.common.processor.MapProcessor` with the
    provided field configuration.

    :param mapprocessor_config: Configuration fields for
        `:class:`~CHAP.common.processor.MapProcessor`.
    :type mapprocessor_config: dict
    :returns: Return value of
        `CHAP.common.processor.MapProcessor.process`
    :rtype: NXentry
    """
    proc = MapProcessor(**mapprocessor_config, logger=logger)
    return proc.process([])


def write_map_nxentry(nexuswriter_config, map_nxentry):
    """Run :class:`CHAP.common.writer.NexusWriter` to write the map
    NXentry to a NeXus file.

    :param nexuswriter_config: Configuration fields for
        :class:`CHAP.common.writer.NexusWriter` (``filename``,
        ``nxpath``, ``force_overwrite``).
    :type nexuswriter_config: dict
    :param map_nxentry: NXentry object to write.
    :type map_nxentry: nexusformat.nexus.NXentry
    """
    data = [
        PipelineData(
            data=map_nxentry
        )
    ]
    writer = NexusWriter(**nexuswriter_config, logger=logger)
    writer.write(data)


class ParseDetectorConfig(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """Parse the ``--detector_config`` argument into a detector
        configuration dictionary.

        Accepts either a path to a YAML file or a JSON string
        representing a valid
        :class:`~CHAP.common.models.map.DetectorConfig`. If ``values``
        is ``None``, falls back to :data:`VORTEX_CONFIG`.

        :param parser: The argument parser.
        :type parser: argparse.ArgumentParser
        :param namespace: The namespace being populated.
        :type namespace: argparse.Namespace
        :param values: The raw string value supplied on the command
            line (file path, JSON string, or ``None``).
        :type values: str or None
        :param option_string: The option string used to invoke this
            action, if any.
        :type option_string: str or None
        :raises argparse.ArgumentError: If ``values`` is a non-``None``
            non-string value.
        """
        if os.path.isfile(values):
            # Try loading detector_config from .yaml file
            with open(values, 'r') as inf:
                detector_config = yaml.load(inf)
        elif isinstance(values, str):
            # Try loading detector_config from JSON string
            detector_config = json.loads(values)
        elif values is not None:
            raise argparse.ArgumentError(
                self, f'Invalid value "{values}"'
            )
        else:
            # Fallback: use VORTEX_CONFIG
            setattr(namespace, self.dest, VORTEX_CONFIG)


def run():
    """Entry point for the XRF raw-data CLI.

    Parses command-line arguments, builds a
    :class:`~CHAP.common.models.map.MapConfig` from the specified SPEC
    scan, runs :class:`~CHAP.common.processor.MapProcessor` to produce
    an NXentry, and writes the result to a NeXus file via
    :class:`~CHAP.common.writer.NexusWriter`.
    """
    parser = argparse.ArgumentParser(
        description='''CLI to convert raw data from an FMB-style XRF
        scan into NeXus format.'''
    )
    parser.add_argument(
        '--spec_file',
        type=str,
        required=True,
        help='''Path to the SPEC file containing the scan data.''',
    )
    parser.add_argument(
        '--scan_number',
        type=int,
        required=True,
        help='''Scan number within the SPEC file to process.''',
    )
    parser.add_argument(
        '--station',
        choices=['id1a3', 'id3a', 'id3b', 'id4b'],
        required=False,
        default='id3b',
        help='''CHESS station identifier. Default: id3b.''',
    )
    parser.add_argument(
        '--experiment',
        choices=['EDD', 'GIWAXS', 'HDRM', 'SAXSWAXS', 'TOMO', 'XRF'],
        required=False,
        default='XRF',
        help='''Experiment type. Default: XRF.''',
    )
    parser.add_argument(
        '--dwell_time_actual_counter_name',
        type=str,
        required=False,
        default='mcs0',
        help='''Name of the SPEC counter recording actual dwell
        time. Default: mcs0.''',
    )
    parser.add_argument(
        '--presample_intensity_counter_name',
        type=str,
        required=False,
        default='ic1',
        help='''Name of the SPEC counter recording pre-sample beam
        intensity. Default: ic1.''',
    )
    parser.add_argument(
        '--postsample_intensity_counter_name',
        type=str,
        required=False,
        default='diode',
        help='''Name of the SPEC counter recording post-sample beam
        intensity. Default: diode.''',
    )
    parser.add_argument(
        '--detector_config',
        type=str,
        required=False,
        default=VORTEX_CONFIG,
        action=ParseDetectorConfig,
        help='''Path to YAML file containing a
        `CHAP.common.models.map.DetectorConfig` configuration, or a
        JSON string containing the same.''',
    )
    parser.add_argument(
        '--outfilename',
        type=str,
        required=True,
        help='''Path to the output NeXus (.nxs) file.''',
    )
    parser.add_argument(
        '--outnxpath',
        type=str,
        required=False,
        default=None,
        help='''NeXus path within the output file at which to write
             the NXentry. If omitted, the writer uses its default
             path.''',
    )
    parser.add_argument(
        '--force_overwrite',
        action='store_true',
        help='''Overwrite the output file if it already exists.''',
    )
    
    args = parser.parse_args(sys.argv[1:])

    logger.info(f'Using detector configuration {args.detector_config}')
    map_config = get_map_config(args)
    logger.info(f'Using map configuration {map_config}')
    mapprocessor_config = {
        'config': map_config,
        'detector_config': args.detector_config,
    }
    map_nxentry = process_map_config(mapprocessor_config)
    logger.info(f'Got map nxentry:\n{map_nxentry.tree}')

    nexuswriter_config = {
        'filename': args.outfilename,
        'nxpath': args.outnxpath,
        'force_overwrite': args.force_overwrite,
    }
    logger.info(f'Writing map nxentry to {nexuswriter_config}')
    write_map_nxentry(nexuswriter_config, map_nxentry)
