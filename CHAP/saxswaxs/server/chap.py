"""CHAP processing code using "cached" CHAP ``PipelineItem``s for
better performance."""

from CHAP.common.map_utils import SpecScanToMapConfigProcessor
from CHAP.common.reader import YAMLReader
from CHAP.common.writer import YAMLWriter, ZarrWriter, ZarrValuesWriter
from CHAP.pipeline import PipelineData
from CHAP.saxswaxs.processor import SetupProcessor, UpdateValuesProcessor
from functools import cache
from pathlib import Path
import subprocess
from typing import Optional

from pydantic import BaseModel, ConfigDict

from CHAP.saxswaxs.server import get_logger
from CHAP.saxswaxs.server.saxswaxs_to_chap import (
    saxswaxs_to_chap,
    make_pipeline as _make_pipeline,
    convert_configs as _convert_configs,
)

# functions whose caches will need to be cleared regularly to work
# with live data processing
from CHAP.common.models.map import get_scanparser
from chess_scanparsers.scanparsers import (
    filespec,
    list_fmb_saxswaxs_detector_files,
)

logger = get_logger('chap')

def cache_clear():
    """Clear scan parser and file listing caches before processing new data."""
    get_scanparser.cache_clear()
    filespec.cache_clear()
    list_fmb_saxswaxs_detector_files.cache_clear()


def init():
    """No-op placeholder for module initialization."""
    pass


@cache
def _read_yaml(filename, schema):
    """Read a YAML config file and return it wrapped in a :class:`PipelineData` object.

    Results are cached so the same file is only read once per process lifetime.
    Call :func:`cache_clear` before processing new data to invalidate stale entries.

    :param filename: Path to the YAML file to read.
    :type filename: str or Path
    :param schema: CHAP schema string used to validate and parse the YAML contents.
    :type schema: str
    :returns: Parsed config wrapped in a PipelineData container.
    :rtype: PipelineData
    """
    return PipelineData(
        name='YAMLReader',
        data=YAMLReader.run(
            filename=str(filename),
            schema=schema,
        ),
        schema=schema,
    )


@cache
def read_configs(detectors_yaml, map_yaml, pyfai_yaml, corrections_yaml, fits_yaml):
    """Read the four config YAML files required for SAXS/WAXS processing.

    Results are cached; call :func:`cache_clear` before processing a new scan.

    :param detectors_yaml: Path to the detector config YAML file.
    :type detectors_yaml: Path
    :param map_yaml: Path to the map config YAML file.
    :type map_yaml: Path
    :param pyfai_yaml: Path to the pyFAI integration processor config YAML file.
    :type pyfai_yaml: Path
    :param corrections_yaml: Path to the corrections config YAML file.
    :type corrections_yaml: Path
    :returns: List of four PipelineData objects for detector, map, pyFAI, and
        corrections configs respectively.
    :rtype: list[PipelineData]
    """
    return [
        _read_yaml(
            detectors_yaml,
            'common.models.map.DetectorConfig'
        ),
        _read_yaml(
            map_yaml,
            'common.models.map.MapConfig'
        ),
        _read_yaml(
            pyfai_yaml,
            'common.models.integration.PyfaiIntegrationConfig'
        ),
        _read_yaml(
            corrections_yaml,
            'saxswaxs.models.CorrectionsConfig'
        ),
        _read_yaml(
            fits_yaml,
            'saxswaxs.models.FitsConfig'
        ),
    ]


def setup(cfg):
    """Run the CHAP setup pipeline to create the Zarr dataset structure.

    Generates map and pipeline config files from the spec scan, reads all
    config files, runs :class:`SetupProcessor` to create the Zarr dataset
    structure, and writes the result to disk.

    :param cfg: Configuration for the setup task.
    :type cfg: SetupCfg
    """
    cache_clear()
    setup_configs(cfg)
    logger.info('Reading')
    data = read_configs(
        cfg.detectors_yaml, cfg.map_yaml, cfg.pyfai_yaml, cfg.corrections_yaml, cfg.fits_yaml,
    )
    logger.info('Processing')
    zarr_tree = [
        PipelineData(
            data=SetupProcessor.run(
                data=data,
                dataset_chunks=cfg.dataset_chunks,
                raw_data=False,
            ),
            name='saxswaxs.processor.SetupProcessor.run',
        )
    ]
    logger.info('Writing')
    ZarrWriter.run(
        data=zarr_tree,
        filename=str(cfg.data_zarr),
        force_overwrite=True
    )


def update(cfg):
    """Run the CHAP update pipeline to process a scan's data into the Zarr dataset.

    Reads all config files, runs :class:`UpdateValuesProcessor` for the specified
    scan and index slice, and writes the resulting values into the existing Zarr
    dataset.

    :param cfg: Configuration for the update task.
    :type cfg: UpdateCfg
    """
    cache_clear()
    logger.info('Reading')
    data = read_configs(
        cfg.detectors_yaml, cfg.map_yaml, cfg.pyfai_yaml, cfg.corrections_yaml, cfg.fits_yaml,
    )
    logger.info('Processing')
    values = [
        PipelineData(
            data=UpdateValuesProcessor.run(
                data=data,
                filename=str(cfg.data_zarr),
                spec_file=cfg.spec_file,
                scan_number=cfg.scan_number,
                idx_slice=dict(
                    start=cfg.idx_slice_start,
                    stop=cfg.idx_slice_stop,
                    step=cfg.idx_slice_step,
                ),
                raw_data=True,
            ),
            name='UpdateValuesProcessor.run',
        )
    ]
    logger.info('Writing')
    ZarrValuesWriter.run(
        data=values,
        filename=str(cfg.data_zarr),
        resize_axis=0,
        idx_slice=dict(
            start=cfg.idx_slice_start,
            stop=cfg.idx_slice_stop,
            step=cfg.idx_slice_step,
        ),
        force_overwrite=True,
    )


def convert(cfg):
    """Run the CHAP convert pipeline to convert the Zarr dataset to NeXus format.

    Launches CHAP as a subprocess with the ``convert`` pipeline defined in
    ``cfg.outputdir/pipeline.yaml``. Subprocess output is written to
    ``cfg.outputdir/chap_convert.log``.

    :param cfg: Configuration for the convert task.
    :type cfg: ConvertCfg
    """
    logger.info("CHAP convert starting")

    logname = cfg.outputdir / "chap_convert.log"
    with open(logname, "w") as logfile:
        process = subprocess.Popen(
            [
                "CHAP",
                cfg.outputdir / "pipeline.yaml",
                "-p",
                "convert",
            ],
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )
        process.wait()
    logger.info(f"CHAP convert logging to {logname}")


def setup_configs(cfg):
    """Write map config and CHAP pipeline config YAML files for a spec scan.

    Calls :func:`scan_to_map` to generate the map config from the spec scan,
    then :func:`saxswaxs_to_chap` to generate the corresponding CHAP pipeline
    config files in the output directory.

    :param cfg: Configuration containing spec file, scan number, counter names,
        and output file paths.
    :type cfg: SetupCfg
    """
    map_config = PipelineData(
        data=SpecScanToMapConfigProcessor.run(
            spec_file=cfg.spec_file,
            scan_number=cfg.scan_number,
            station="id3b",
            experiment="SAXSWAXS",
            dwell_time_actual_counter_name=cfg.dwell_time_actual_counter_name,
            presample_intensity_counter_name=cfg.presample_intensity_counter_name,
            postsample_intensity_counter_name=cfg.postsample_intensity_counter_name,
            validate_data_present=False,
        ),
    )
    YAMLWriter.run(
        data=map_config,
        filename=cfg.map_yaml,
    )

    logger.info(
        f"saxswaxs_to_chap({cfg.map_yaml}, {cfg.tool_yamls}, {cfg.outputdir})"
    )
    saxswaxs_to_chap(
        str(cfg.map_yaml), [str(t_y) for t_y in cfg.tool_yamls], str(cfg.outputdir),
        detector_filename=str(cfg.detectors_yaml),
        pyfai_filename=str(cfg.pyfai_yaml),
        correction_filename=str(cfg.corrections_yaml),
        fits_filename=str(cfg.fits_yaml),
    )


class SaxswaxsCfg(BaseModel):
    """Base configuration shared by all SAXS/WAXS processing tasks.

    Contains paths to the spec file, all four YAML config files (detector,
    map, pyFAI integration, corrections), and the output Zarr dataset.

    :ivar spec_file: Path to the SPEC file containing the scan.
    :vartype spec_file: Path
    :ivar scan_number: Number of the scan within the SPEC file.
    :vartype scan_number: int
    :ivar detectors_yaml: Path to the detector config YAML file.
    :vartype detectors_yaml: Path
    :ivar map_yaml: Path to the map config YAML file.
    :vartype map_yaml: Path
    :ivar pyfai_yaml: Path to the pyFAI integration processor config YAML file.
    :vartype pyfai_yaml: Path
    :ivar corrections_yaml: Path to the corrections config YAML file.
    :vartype corrections_yaml: Path
    :ivar data_zarr: Path to the output Zarr dataset.
    :vartype data_zarr: Path
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec_file: Path
    scan_number: int

    detectors_yaml: Path
    map_yaml: Path
    pyfai_yaml: Path
    corrections_yaml: Path
    fits_yaml: Path

    data_zarr: Path


class SetupCfg(SaxswaxsCfg):
    """Configuration for the setup task, which creates the Zarr dataset structure.

    Extends :class:`SaxswaxsCfg` with the output directory, tool config files,
    SPEC counter names for intensity and dwell time, and dataset chunk sizes.

    :ivar outputdir: Directory for output CHAP config files and the Zarr dataset.
    :vartype outputdir: Path
    :ivar tool_yamls: List of tool config YAML file paths.
    :vartype tool_yamls: list[Path]
    :ivar dwell_time_actual_counter_name: SPEC counter column name for actual dwell times.
    :vartype dwell_time_actual_counter_name: str
    :ivar presample_intensity_counter_name: SPEC counter column name for presample intensity.
    :vartype presample_intensity_counter_name: str
    :ivar postsample_intensity_counter_name: SPEC counter column name for postsample
        intensity, or ``None`` if not recorded.
    :vartype postsample_intensity_counter_name: str or None
    :ivar dataset_chunks: Chunk sizes for the Zarr dataset dimensions.
    :vartype dataset_chunks: list[int]
    """

    outputdir: Path

    tool_yamls: list[Path]

    dwell_time_actual_counter_name: str
    presample_intensity_counter_name: str
    postsample_intensity_counter_name: Optional[str] = None

    dataset_chunks: list[int]


class UpdateCfg(SaxswaxsCfg):
    """Configuration for the update task, which fills data into the Zarr dataset.

    Extends :class:`SaxswaxsCfg` with an index slice identifying which rows
    of the dataset to process in this update.

    :ivar idx_slice_start: Start index of the row slice to process, defaults to ``0``.
    :vartype idx_slice_start: int
    :ivar idx_slice_stop: Stop index of the row slice to process, defaults to ``-1``.
    :vartype idx_slice_stop: int
    :ivar idx_slice_step: Step of the row slice to process, defaults to ``1``.
    :vartype idx_slice_step: int
    """

    idx_slice_start: int = 0
    idx_slice_stop: int = -1
    idx_slice_step: int = 1


class ConvertCfg(BaseModel):
    """Configuration for the convert task, which converts the Zarr dataset to NeXus.

    Only requires the output directory containing the ``pipeline.yaml`` written
    by the setup task.

    :ivar outputdir: Directory containing the CHAP ``pipeline.yaml`` and Zarr dataset.
    :vartype outputdir: Path
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    outputdir: Path


class ConfigFilesCfg(BaseModel):
    """Base configuration shared by tasks that read and write CHAP config files
    in a common output directory.

    :ivar outputdir: Directory to which output config files will be written, and
        against which relative filenames are resolved.
    :vartype outputdir: Path
    :ivar detector_filename: Path to the detector config YAML file. If relative,
        resolved against ``outputdir``. Defaults to ``'detector_config.yaml'``.
    :vartype detector_filename: str
    :ivar pyfai_filename: Path to the pyFAI integration processor config YAML
        file. If relative, resolved against ``outputdir``. Defaults to
        ``'pyfai_integration_processor_config.yaml'``.
    :vartype pyfai_filename: str
    :ivar correction_filename: Path to the corrections config YAML file. If
        relative, resolved against ``outputdir``. Defaults to
        ``'corrections_config.yaml'``.
    :vartype correction_filename: str
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    outputdir: Path
    detector_filename: str = 'detector_config.yaml'
    pyfai_filename: str = 'pyfai_integration_processor_config.yaml'
    correction_filename: str = 'corrections_config.yaml'


class MakePipelineCfg(ConfigFilesCfg):
    """Configuration for the make_pipeline task, which writes a ``pipeline.yaml``
    from pre-existing config files without requiring the old workflow library.

    Extends :class:`ConfigFilesCfg` with the map, fits, and pipeline filenames.

    :ivar map_filename: Path to the map config YAML file. If relative, resolved
        against ``outputdir``. Defaults to ``'map_config.yaml'``.
    :vartype map_filename: str
    :ivar fits_filename: Path to the fits config YAML file. If relative, resolved
        against ``outputdir``. Defaults to ``'fits_config.yaml'``.
    :vartype fits_filename: str
    :ivar pipeline_filename: Output filename for the pipeline YAML. Defaults to
        ``'pipeline.yaml'``.
    :vartype pipeline_filename: str
    """

    map_filename: str = 'map_config.yaml'
    fits_filename: str = 'fits_config.yaml'
    pipeline_filename: str = 'pipeline.yaml'


class ConvertConfigsCfg(ConfigFilesCfg):
    """Configuration for the convert_configs task, which writes detector,
    pyFAI integration, and corrections config YAML files from old-style
    saxswaxs workflow tool config files.

    Extends :class:`ConfigFilesCfg` with the list of tool YAML files to convert.

    :ivar tool_yamls: List of tool config YAML file paths to convert.
    :vartype tool_yamls: list[Path]
    """

    tool_yamls: list[Path]


def make_pipeline(cfg):
    """Run the make_pipeline task to write a ``pipeline.yaml`` from pre-existing
    config files.

    Calls :func:`CHAP.saxswaxs.server.saxswaxs_to_chap.make_pipeline` with the paths
    and filenames from ``cfg``.

    :param cfg: Configuration for the make_pipeline task.
    :type cfg: MakePipelineCfg
    """
    _make_pipeline(
        str(cfg.outputdir),
        map_filename=cfg.map_filename,
        detector_filename=cfg.detector_filename,
        pyfai_filename=cfg.pyfai_filename,
        correction_filename=cfg.correction_filename,
        fits_filename=cfg.fits_filename,
        pipeline_filename=cfg.pipeline_filename,
    )


def convert_configs(cfg):
    """Run the convert_configs task to write detector, pyFAI integration, and
    corrections config YAML files from old-style saxswaxs workflow tool configs.

    Calls :func:`CHAP.saxswaxs.server.saxswaxs_to_chap.convert_configs` with the paths
    and filenames from ``cfg``.

    :param cfg: Configuration for the convert_configs task.
    :type cfg: ConvertConfigsCfg
    """
    _convert_configs(
        str(cfg.outputdir),
        [str(t) for t in cfg.tool_yamls],
        detector_filename=cfg.detector_filename,
        pyfai_filename=cfg.pyfai_filename,
        correction_filename=cfg.correction_filename,
    )
