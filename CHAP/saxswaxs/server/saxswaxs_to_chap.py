
"""Script to act as a drop-in replacement for the old saxswaxsworkflow
CLI tool.
Instead of running the old workflow code, a new CHAP pipeline will be
constructed & run from the given tool & map configs.
"""

import argparse
import logging
import os
import sys

import yaml


class VerboseSafeDumper(yaml.SafeDumper):
    """YAML SafeDumper subclass that writes all nodes explicitly without aliases."""

    def ignore_aliases(self, data):
        """Return True to prevent YAML from using anchor/alias references."""
        return True


def saxswaxs_to_chap(
        map_config_file, tool_config_files, outputdir,
        detector_filename='detector_config.yaml',
        pyfai_filename='pyfai_integration_processor_config.yaml',
        correction_filename='corrections_config.yaml',
        fits_filename='fits_config.yaml',
        pipeline_filename='pipeline.yaml'):
    """Build CHAP pipeline config files from old-style saxswaxs workflow configs.

    Loads the map and tool config files using the old ``workflow`` library, then
    calls :func:`wf_to_chap` to write the corresponding CHAP pipeline YAML files
    to ``outputdir``.

    :param map_config_file: Path to the map config YAML file.
    :type map_config_file: str
    :param tool_config_files: List of tool config YAML file paths.
    :type tool_config_files: list[str]
    :param outputdir: Directory in which to write the output CHAP config files.
    :type outputdir: str
    :param detector_filename: Output filename for the detector config YAML,
        defaults to ``'detector_config.yaml'``.
    :type detector_filename: str, optional
    :param pyfai_filename: Output filename for the pyFAI integration processor
        config YAML, defaults to ``'pyfai_integration_processor_config.yaml'``.
    :type pyfai_filename: str, optional
    :param correction_filename: Output filename for the corrections config YAML,
        defaults to ``'corrections_config.yaml'``.
    :type correction_filename: str, optional
    :param pipeline_filename: Output filename for the CHAP pipeline config YAML,
        defaults to ``'pipeline.yaml'``.
    :type pipeline_filename: str, optional
    """
    # Initialize old-style saxswaxs workflow configuration objects
    from workflow.map import MapConfig
    from workflow.basemodel import BaseModel
    from workflow.workflow import Workflow

    logger = logging.getLogger(__name__)
    map_config = MapConfig.construct_from_file(
        map_config_file,
        logger=logger, validate_data_present=False)
    tools = [
        BaseModel.construct_from_file(
            tool_config_file, logger=logger)
        for tool_config_file in tool_config_files
    ]
    wf = Workflow(map_config=map_config, tools=tools,
                  validate_data_present=False)

    # Compose chap pipeline config form old-style saxswaxs workflow
    chap_config = wf_to_chap(
        wf, outputdir,
        map_filename=map_config_file,
        detector_filename=detector_filename,
        pyfai_filename=pyfai_filename,
        correction_filename=correction_filename,
        fits_filename=fits_filename,
        pipeline_filename=pipeline_filename,
    )


def make_pipeline(outputdir,
                  map_filename='map_config.yaml',
                  detector_filename='detector_config.yaml',
                  pyfai_filename='pyfai_integration_processor_config.yaml',
                  correction_filename='corrections_config.yaml',
                  fits_filename='fits_config.yaml',
                  pipeline_filename='pipeline.yaml'):
    """Compose a pipeline file for a complete saxswaxs workflow based
    on the config files provided, and asssuming they all already
    exist. Sort of a lightweight version of wf_to_chap."""
    from CHAP.common.reader import YAMLReader

    outputdir = os.path.abspath(outputdir)

    if not os.path.isabs(map_filename):
        map_filename = os.path.join(outputdir, map_filename)
    if not os.path.isabs(detector_filename):
        detector_filename = os.path.join(outputdir, detector_filename)
    if not os.path.isabs(pyfai_filename):
        pyfai_filename = os.path.join(outputdir, pyfai_filename)
    if not os.path.isabs(correction_filename):
        correction_filename = os.path.join(outputdir, correction_filename)
    if not os.path.isabs(fits_filename):
        fits_filename = os.path.join(outputdir, fits_filename)

    map_config = YAMLReader.run(
        filename=map_filename, schema='common.models.map.MapConfig')
    zarr_filename = f'{map_config.title}.zarr'
    nxs_filename = f'{map_config.title}.nxs'

    readers = [
        {
            'common.reader.YAMLReader': {
                'filename': detector_filename,
                'schema': 'common.models.map.DetectorConfig'
            }
        },
        {
            'common.reader.YAMLReader': {
                'filename': map_filename,
                'schema': 'common.models.map.MapConfig'
            }
        },
        {
            'common.reader.YAMLReader': {
                'filename': pyfai_filename,
                'schema': 'common.models.integration.PyfaiIntegrationConfig'
            }
        },
        {
            'common.reader.YAMLReader': {
                'filename': correction_filename,
                'schema': 'saxswaxs.models.CorrectionsConfig'
            }
        },
        {
            'common.reader.YAMLReader': {
                'filename': fits_filename,
                'schema': 'saxswaxs.models.FitsConfig'
            }
        },
    ]

    update_pipelines = {}
    npts = 0
    nrows = 0
    row_npts = 1
    for scans in map_config.spec_scans:
        for scan_number in scans.scan_numbers:
            sp = scans.get_scanparser(scan_number)
            _npts = int(sp.spec_scan_npts)
            if len(sp.spec_scan_shape) > 1:
                _nrows = sp.spec_scan_shape[1]
                row_npts = sp.spec_scan_shape[0]
            else:
                _nrows = 1
                row_npts = _npts
            for i in range(_nrows):
                idx_slice = {
                    'start': npts + (i * row_npts),
                    'stop': npts + (i * row_npts) + row_npts,
                    'step': 1,
                }
                update_pipelines[f'update_{nrows + i}'] = [
                    *readers,
                    {
                        'saxswaxs.processor.UpdateValuesProcessor': {
                            'raw_data': False,
                            'filename': zarr_filename,
                            'spec_file': scans.spec_file,
                            'scan_number': scan_number,
                            'idx_slice': idx_slice,
                        }
                    },
                    {
                        'common.ZarrValuesWriter': {
                            'filename': zarr_filename,
                            'resize_axis': 0,
                            'idx_slice': idx_slice,
                            'force_overwrite': True,
                        }
                    }
                ]
            nrows += _nrows
            npts += _npts

    chap_config = {
        'config': {
            'root': outputdir,
            'log_level': 'debug',
        },
        'setup': [
            *readers,
            {
                'saxswaxs.processor.SetupProcessor': {
                    'raw_data': False,
                    'dataset_chunks': [row_npts],
                }
            },
            {
                'common.writer.ZarrWriter': {
                    'filename': zarr_filename,
                    'force_overwrite': True,
                }
            }
        ],
        **update_pipelines,
        'convert': [
            {
                'common.processor.ZarrToNexusProcessor': {
                    'zarr_filename': zarr_filename,
                    'nexus_filename': nxs_filename,
                }
            }
        ]
    }
    pipeline_path = os.path.join(outputdir, pipeline_filename)
    print(f'Writing to {pipeline_path}')
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    with open(pipeline_path, 'w') as outf:
        yaml.dump(chap_config, outf, sort_keys=False, Dumper=VerboseSafeDumper)
    return chap_config


def convert_configs(outputdir, tool_config_files,
                    detector_filename='detector_config.yaml',
                    pyfai_filename='pyfai_integration_processor_config.yaml',
                    correction_filename='corrections_config.yaml'):
    """Write the new CHAP.saxswaxs-formatted pyfai and corrections
    config files based on the old workflow tool files provided. Should
    be independent from any map configuration or
    saxswaxsworkflow.workflow.Workflow object -- use the dictionaries
    in the tool_config_files only."""

    outputdir = os.path.abspath(outputdir)

    pyfai_integration_processor_config = {
        'azimuthal_integrators': [],
        'integrations': [],
    }
    detectors = []
    corrections = []
    for tool_config_file in tool_config_files:
        with open(tool_config_file) as f:
            t = yaml.safe_load(f)
        if t.get('tool_type') == 'integration':
            # Build azimuthal_integrators entries
            for det in t.get('detectors', []):
                prefix = det['prefix']
                already_added = any(
                    ai['id'] == prefix
                    and ai['poni_file'] == str(det['poni_file'])
                    and ai['mask_file'] == str(det['mask_file'])
                    for ai in pyfai_integration_processor_config['azimuthal_integrators']
                )
                if not already_added:
                    pyfai_integration_processor_config['azimuthal_integrators'].append(
                        {
                            'id': prefix,
                            'poni_file': str(det['poni_file']),
                            'mask_file': str(det['mask_file']),
                        }
                    )
                # Build detector_config entries
                visited = any(prefix == d['id'] for d in detectors)
                if not visited:
                    placeholder_shape = (1, 1)
                    if prefix == 'PIL5':
                        shape = [619, 487]
                    elif prefix in ('PIL9', 'PIL11'):
                        shape = [407, 487]
                    else:
                        print(
                            f'WARNING: unrecognized detector prefix {prefix}; '
                            + f'using placeholder shape {placeholder_shape}'
                        )
                        shape = list(placeholder_shape)
                    detectors.append({'id': prefix, 'shape': shape})

            # Build integrations entry
            integration_config = {'name': t['title']}
            if t.get('integration_type') == 'radial':
                integration_config['integration_method'] = 'integrate_radial'
                integration_config['integration_params'] = {
                    'ais': [det['prefix'] for det in t.get('detectors', [])],
                    'npt': t['azimuthal_npt'],
                    'npt_rad': t['radial_npt'],
                    'radial_range': [t['radial_min'], t['radial_max']],
                    'azimuth_range': [t['azimuthal_min'], t['azimuthal_max']],
                    'unit': t['azimuthal_units'],
                    'radial_unit': t['radial_units'],
                    'method': 'bbox_csr_cython',
                }
            else:
                integration_config['multi_geometry'] = {
                    'ais': [det['prefix'] for det in t.get('detectors', [])],
                    'unit': t['radial_units'],
                    'radial_range': [t['radial_min'], t['radial_max']],
                    'azimuth_range': [t['azimuthal_min'], t['azimuthal_max']],
                }
                if t.get('integration_type') == 'azimuthal':
                    integration_config['integration_method'] = 'integrate1d'
                    integration_config['integration_params'] = {
                        'npt': t['radial_npt'],
                        'method': 'bbox_csr_cython',
                    }
                elif t.get('integration_type') == 'cake':
                    integration_config['integration_method'] = 'integrate2d'
                    integration_config['integration_params'] = {
                        'npt_rad': t['radial_npt'],
                        'npt_azim': t['azimuthal_npt'],
                        'method': 'bbox_csr_cython',
                    }
            pyfai_integration_processor_config['integrations'].append(integration_config)
        else:
            # It's a corrections tool — include all fields except tool_type
            correction = {k: v for k, v in t.items()
                          if k not in ('tool_type', 'validate_data_present')}
            corrections.append(correction)

    # Write detector config .yaml
    detector_config = {'detectors': detectors}
    if not os.path.isabs(detector_filename):
        detector_filename = os.path.join(outputdir, detector_filename)
    print(f'Writing to {detector_filename}')
    os.makedirs(os.path.dirname(detector_filename), exist_ok=True)
    with open(detector_filename, 'w') as outf:
        yaml.dump(detector_config, outf, sort_keys=False,
                  Dumper=VerboseSafeDumper)

    # Write pyfai config .yaml
    if not os.path.isabs(pyfai_filename):
        pyfai_filename = os.path.join(outputdir, pyfai_filename)
    print(f'Writing to {pyfai_filename}')
    os.makedirs(os.path.dirname(pyfai_filename), exist_ok=True)
    with open(pyfai_filename, 'w') as outf:
        yaml.dump(pyfai_integration_processor_config, outf, sort_keys=False,
                  Dumper=VerboseSafeDumper)

    # Write corrections config .yaml
    correction_config = {'corrections': corrections}
    if not os.path.isabs(correction_filename):
        correction_filename = os.path.join(outputdir, correction_filename)
    print(f'Writing to {correction_filename}')
    os.makedirs(os.path.dirname(correction_filename), exist_ok=True)
    with open(correction_filename, 'w') as outf:
        yaml.dump(correction_config, outf, sort_keys=False,
                  Dumper=VerboseSafeDumper)


def wf_to_chap(wf, outputdir,
               map_filename='map_config.yaml',
               detector_filename='detector_config.yaml',
               pyfai_filename='pyfai_integration_processor_config.yaml',
               correction_filename='corrections_config.yaml',
               fits_filename='fits_config.yaml',
               pipeline_filename='pipeline.yaml'):
    """Convert an old-style SAXSWAXS Workflow configuration into the
    analogous CHAP pipeline configuration.

    Writes detector, pyFAI integration, corrections, map, and pipeline YAML
    config files to ``outputdir`` and returns the pipeline config dict.

    :param wf: Workflow configuration to convert.
    :type wf: workflow.Workflow
    :param outputdir: Directory to which all output config .yaml files will
        be written.
    :type outputdir: str
    :param map_filename: Filename for the map config .yaml file, defaults to
        ``'map_config.yaml'``.
    :type map_filename: str, optional
    :param detector_filename: Filename for the detector config .yaml file,
        defaults to ``'detector_config.yaml'``.
    :type detector_filename: str, optional
    :param pyfai_filename: Filename for the PyfaiIntegrationProcessorConfig
        .yaml file, defaults to ``'pyfai_integration_processor_config.yaml'``.
    :type pyfai_filename: str, optional
    :param correction_filename: Filename for the corrections config .yaml file,
        defaults to ``'corrections_config.yaml'``.
    :type correction_filename: str, optional
    :param pipeline_filename: Filename for the CHAP pipeline config .yaml file,
        defaults to ``'pipeline.yaml'``.
    :type pipeline_filename: str, optional
    :returns: Full CHAP pipeline config dict (setup, per-row update, and convert
        pipelines).
    :rtype: dict
    """
    outputdir = os.path.abspath(outputdir)

    # Write map configuration to file
    map_config = wf.map_config
    if not os.path.isabs(map_filename):
        map_filename = os.path.join(outputdir, map_filename)

    # Write pyfai integration processor config to file
    pyfai_integration_processor_config = {
        'azimuthal_integrators': [],
        'integrations': [],
    }
    detectors = []
    corrections = []
    for t in wf.tools:
        if hasattr(t, 'detectors'):
            # It's an integration tool.
            add_integration(pyfai_integration_processor_config, t)
            for det in t.detectors:
                visited = any([det.prefix == d['id'] for d in detectors])
                if not visited:
                    # Decide on detector shape (use list not tuple for
                    # yaml file)
                    placeholder_shape = (1, 1)
                    if det.prefix == 'PIL5':
                        shape = [619, 487]
                    elif det.prefix in ('PIL9', 'PIL11'):
                        shape = [407, 487]
                    else:
                        print(
                            f'WARNING: unrecorgnized detector prefix {det.prefix}; '
                            + f'using placeholder shape {placeholder_shape}'
                        )
                        shape = placeholder_shape
                    detectors.append(
                        {
                            'id': det.prefix,
                            'shape': shape,
                        }
                    )
        else:
            # It's a corrections tool
            corrections.append(
                t.model_dump(
                    mode='json',
                    exclude_unset=True,
                    exclude_defaults=True,
                    exclude=['validate_data_present'],
                )
            )

    # Write detector config .yaml
    detector_config = {'detectors': detectors}
    if not os.path.isabs(detector_filename):
        detector_filename = os.path.join(outputdir, detector_filename)
    print(f'Writing to {detector_filename}')
    os.makedirs(os.path.dirname(detector_filename), exist_ok=True)
    with open(detector_filename, 'w') as outf:
        yaml.dump(detector_config, outf, sort_keys=False,
                  Dumper=VerboseSafeDumper)

    # Write pyfai config .yaml
    if not os.path.isabs(pyfai_filename):
        pyfai_filename = os.path.join(outputdir, pyfai_filename)
    print(f'Writing to {pyfai_filename}')
    os.makedirs(os.path.dirname(pyfai_filename), exist_ok=True)
    with open(pyfai_filename, 'w') as outf:
        yaml.dump(pyfai_integration_processor_config, outf, sort_keys=False,
                  Dumper=VerboseSafeDumper)

    # Write corrections config .yaml
    correction_config = {'corrections': corrections}
    if not os.path.isabs(correction_filename):
        correction_filename = os.path.join(outputdir, correction_filename)
    print(f'Writing to {correction_filename}')
    os.makedirs(os.path.dirname(correction_filename), exist_ok=True)
    with open(correction_filename, 'w') as outf:
        yaml.dump(correction_config, outf, sort_keys=False,
                  Dumper=VerboseSafeDumper)

    # Iterate through each scan in the map and compose the individual
    # scan-row-wise update jobs so the dataset can be processed in
    # parallel by running all update pipelines at once, or
    # incrementally by running each update pipeline one at a time.
    update_pipelines = {}
    npts = 0
    nrows = 0
    zarr_filename = f'{wf.map_config.title}.zarr'
    nxs_filename = f'{wf.map_config.title}.nxs'
    for scans in map_config.spec_scans:
        for scan_number in scans.scan_numbers:
            sp = scans.get_scanparser(scan_number)
            _npts = int(sp.spec_scan_npts)
            if len(sp.spec_scan_shape) > 1:
                _nrows = sp.spec_scan_shape[1]
                row_npts = sp.spec_scan_shape[0]
            else:
                _nrows = 1
                row_npts = _npts
            for i in range(_nrows):
                idx_slice = {
                    'start': npts + (i * row_npts),
                    'stop': npts + (i * row_npts) + row_npts,
                    'step': 1,
                }
                update_pipelines[f'update_{nrows + i}'] = [
                    {
                        'common.reader.YAMLReader': {
                            'filename': detector_filename,
                            'schema': 'common.models.map.DetectorConfig'
                        }
                    },
                    {
                        'common.reader.YAMLReader': {
                            'filename': map_filename,
                            'schema': 'common.models.map.MapConfig'
                        }
                    },
                    {
                        'common.reader.YAMLReader': {
                            'filename': pyfai_filename,
                            'schema': 'common.models.integration.PyfaiIntegrationConfig'
                        }
                    },
                    {
                        'common.reader.YAMLReader': {
                            'filename': correction_filename,
                            'schema': 'saxswaxs.models.CorrectionsConfig'
                        }
                    },
                    {
                        'common.reader.YAMLReader': {
                            'filename': fits_filename,
                            'schema': 'saxswaxs.models.FitsConfig'
                        }
                    },
                    {
                        'saxswaxs.processor.UpdateValuesProcessor': {
                            'raw_data': False,
                            'filename': zarr_filename,
                            'spec_file': scans.spec_file,
                            'scan_number': scan_number,
                            'idx_slice': idx_slice,
                        }
                    },
                    {
                        'common.ZarrValuesWriter': {
                            'filename': zarr_filename,
                            'resize_axis': 0,
                            'idx_slice': idx_slice,
                            'force_overwrite': True,
                        }
                    }
                ]
            nrows += _nrows
            npts += _npts

    # Compose final CHAP pipeline config and write to file.
    # _restructure_pipeline = restructure_pipeline(wf)
    chap_config = {
        'config': {
            'root': outputdir,
            'log_level': 'debug',
        },
        'setup': [
            {
                'common.reader.YAMLReader': {
                    'filename': detector_filename,
                    'schema': 'common.models.map.DetectorConfig'
                }
            },
            {
                'common.reader.YAMLReader': {
                    'filename': map_filename,
                    'schema': 'common.models.map.MapConfig'
                }
            },
            {
                'common.reader.YAMLReader': {
                    'filename': pyfai_filename,
                    'schema': 'common.models.integration.PyfaiIntegrationConfig'
                }
            },
            {
                'common.reader.YAMLReader': {
                    'filename': correction_filename,
                    'schema': 'saxswaxs.models.CorrectionsConfig'
                }
            },
            {
                'common.reader.YAMLReader': {
                    'filename': fits_filename,
                    'schema': 'saxswaxs.models.FitsConfig'
                }
            },
            {
                'saxswaxs.processor.SetupProcessor': {
                    'raw_data': False,
                    'dataset_chunks': [row_npts],
                }
            },
            {
                'common.writer.ZarrWriter': {
                    'filename': f'{wf.map_config.title}.zarr',
                    'force_overwrite': True
                }
            }
        ],
        **update_pipelines,
        'convert': [
            {
                'common.processor.ZarrToNexusProcessor': {
                    'zarr_filename': zarr_filename,
                    'nexus_filename': nxs_filename,
                }
            }
        ],
        # 'struct': _restructure_pipeline,
    }
    pipeline_filename = os.path.join(outputdir, pipeline_filename)
    print(f'Writing to {pipeline_filename}')
    os.makedirs(os.path.dirname(pipeline_filename), exist_ok=True)
    with open(pipeline_filename, 'w') as outf:
        yaml.dump(chap_config, outf, sort_keys=False, Dumper=VerboseSafeDumper)
    return chap_config


def restructure_pipeline(wf):
    """Build a CHAP pipeline config for restructuring data from unstructured to structured form.

    Reads independent dimension axes and all tool output signals from an existing
    NeXus file and passes them to
    ``saxswaxs.processor.UnstructuredToStructuredProcessor``, writing a structured
    dataset back to the same NeXus file.

    .. note:: This function is not currently used; see the commented-out lines in
        :func:`wf_to_chap`.

    :param wf: Old-style saxswaxs workflow configuration.
    :type wf: workflow.Workflow
    :returns: List of CHAP pipeline step dicts for the restructure pipeline.
    :rtype: list[dict]
    """
    axes = [a.label for a in wf.map_config.independent_dimensions]
    readers = [
        {
            'common.reader.NexusReader': {
                'filename': f'{wf.map_config.title}.nxs',
                'nxpath': f'{wf.map_config.title}/independent_dimensions/{a}',
                'nxmemory': 100000,
                'name': a
            }
        }
        for a in axes]
    fields = [
        {
            'name': a,
            'type': 'axis'
        }
        for a in axes]
    tools = {t.title: t for t in wf.tools}
    for title, tool in tools.items():
        if hasattr(tool, 'integration_type'):
            # Integration tool; only signal is intensity "I"
            signal = 'I'
            _axes = tool.integrated_data_dims
        elif hasattr(tool, 'correction_type'):
            # Corrections tool; varying signals
            signal = 'result'
            _axes = tools[tool.uncorrected_data_title].integrated_data_dims
        readers.extend(
            [
                {
                    'common.reader.NexusReader': {
                        'filename': f'{wf.map_config.title}.nxs',
                        'nxpath': f'{title}/data/{x}',
                        'nxmemory': 100000,
                        'name': f'{title}_{x}'
                    }
                }
                for x in [signal] + _axes]
        )
        fields.extend(
            [
                {
                    'name': f'{title}_{signal}',
                    'type': 'signal',
                    'axes': axes + [f'{title}_{a}' for a in _axes]
                },
                *[
                    {
                        'name': f'{title}_{a}',
                        'type': 'axis'
                    }
                    for a in _axes
                ]
            ]
        )
    pipeline = [
        *readers,
        {
            'saxswaxs.processor.UnstructuredToStructuredProcessor': {
                'fields': fields
            }
        },
        {
            'common.writer.NexusWriter': {
                'filename': f'{wf.map_config.title}.nxs',
                'nxpath': '/structured_data',
                'force_overwrite': True,
            }
        }
    ]
    return pipeline


def add_integration(pyfai_integration_processor_config, tool_config):
    """Add the necessary components from a `workflow.IntegrationTool`
    to an existing config for a
    `saxswaxs.PyfaiIntegrationProcessor`.

    :param pyfai_integration_processor_config: Partial config for
        `saxswaxs.PyfaiIntegrationProcessor` to which the given tool's
        integration will be added.
    :type pyfai_integration_processor_config: dict
    :param tool_config: Old-style workflow integration tool object.
    :type tool_config: workflow.integration.IntegrationConfig
    :returns: Updated `pyfai_integration_processor_config`
    :rtype: dict
    """
    def detector_in_config(det):
        """Convenience function for determining if the given detector
        is already in `pyfai_integration_processor_config`.
        """
        for _det in pyfai_integration_processor_config['azimuthal_integrators']:
            if (_det['id'] == det.prefix
                and _det['poni_file'] == str(det.poni_file)
                and _det['mask_file'] == str(det.mask_file)):
                return True
        return False

    for detector in tool_config.detectors:
        if not detector_in_config(detector):
            pyfai_integration_processor_config['azimuthal_integrators'].append(
                {
                    'id': str(detector.prefix),
                    'poni_file': str(detector.poni_file),
                    'mask_file': str(detector.mask_file),
                }
            )

    integration_config = {
        'name': tool_config.title,
    }
    if tool_config.integration_type == 'radial':
        integration_config['integration_method'] = 'integrate_radial'
        integration_config['integration_params'] = {
            'ais': [det.prefix for det in tool_config.detectors],
            'npt': tool_config.azimuthal_npt,
            'npt_rad': tool_config.radial_npt,
            'radial_range': [tool_config.radial_min, tool_config.radial_max],
            'azimuth_range': [tool_config.azimuthal_min, tool_config.azimuthal_max],
            'unit': tool_config.azimuthal_units,
            'radial_unit': tool_config.radial_units,
            'method': 'bbox_csr_cython',
        }
    else:
        integration_config['multi_geometry'] = {
            'ais': [det.prefix for det in tool_config.detectors],
            'unit': tool_config.radial_units,
            'radial_range': [tool_config.radial_min, tool_config.radial_max],
            'azimuth_range': [tool_config.azimuthal_min, tool_config.azimuthal_max],
        }
        if tool_config.integration_type == 'azimuthal':
            integration_config['integration_method'] = 'integrate1d'
            integration_config['integration_params'] = {
                'npt': tool_config.radial_npt,
                'method': 'bbox_csr_cython',
            }
        elif tool_config.integration_type == 'cake':
            integration_config['integration_method'] = 'integrate2d'
            integration_config['integration_params'] = {
                'npt_rad': tool_config.radial_npt,
                'npt_azim': tool_config.azimuthal_npt,
                'method': 'bbox_csr_cython',
            }
    pyfai_integration_processor_config['integrations'].append(integration_config)

    return pyfai_integration_processor_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--map_config_file', required=True, metavar='map.yaml',
        help='''Map configuration .yaml file to use for constructing
        the workflow.'''
    )
    parser.add_argument(
        '-t', '--tool_config_files', required=True, action='extend',
        nargs='+', metavar='tool.yaml', help='''List of .yaml files
        containing tool configurations to apply to the map
        configuration provided.'''
    )
    parser.add_argument(
        '-f', '--force_overwrite', action='store_true', help='''Use
        this flag to overwrite the output file if it already
        exists.'''
    )
    parser.add_argument(
        '-i', '--inputdir', default='.', help='''Directory containing
        all input files'''
    )
    parser.add_argument(
        '-o', '--outputdir', default='.', help='''Directory in which
        to place output file.'''
    )
    parser.add_argument(
        '-l', '--log', choices=logging._nameToLevel.keys(),
        default='INFO', help='''Specify a preferred logging level.'''
    )
    args = parser.parse_args(sys.argv[1:])

    map_config_file = os.path.join(args.inputdir, args.map_config_file)
    tool_config_files = [
        os.path.join(args.inputdir, tool_config_file)
        for tool_config_file in args.tool_config_files
    ]
    saxswaxs_to_chap(map_config_file, tool_config_files, args.outputdir)
