"""Tests for CHAP.saxswaxs.server.saxswaxs_to_chap."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from CHAP.saxswaxs.server.saxswaxs_to_chap import (
    VerboseSafeDumper,
    convert_configs,
    make_pipeline,
    saxswaxs_to_chap,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Minimal integration tool payload; override individual keys per test.
_INTEGRATION_DEFAULTS = {
    'tool_type': 'integration',
    'title': 'my_integration',
    'integration_type': 'azimuthal',
    'detectors': [
        {'prefix': 'PIL5', 'poni_file': '/cal/det.poni', 'mask_file': '/cal/mask.tif'}
    ],
    'radial_npt': 100,
    'azimuthal_npt': 72,
    'radial_min': 0.1,
    'radial_max': 5.0,
    'azimuthal_min': -180.0,
    'azimuthal_max': 180.0,
    'radial_units': 'q_A^-1',
    'azimuthal_units': 'chi_deg',
}

_CORRECTIONS_DEFAULTS = {
    'tool_type': 'corrections',
    'title': 'my_correction',
    'correction_type': 'transmission',
    'validate_data_present': False,
}


def _write_tool(path, base, **overrides):
    """Write a tool YAML to path, merging overrides into base."""
    data = {**base, **overrides}
    path.write_text(yaml.dump(data))
    return data


def _write_integration(path, **overrides):
    return _write_tool(path, _INTEGRATION_DEFAULTS, **overrides)


def _write_corrections(path, **overrides):
    return _write_tool(path, _CORRECTIONS_DEFAULTS, **overrides)


def _read_outputs(tmp_path, det_name='detector_config.yaml',
                  pyfai_name='pyfai_integration_processor_config.yaml',
                  corr_name='corrections_config.yaml'):
    with open(tmp_path / det_name) as f:
        det = yaml.safe_load(f)
    with open(tmp_path / pyfai_name) as f:
        pyfai = yaml.safe_load(f)
    with open(tmp_path / corr_name) as f:
        corr = yaml.safe_load(f)
    return det, pyfai, corr


def _make_mock_map(title='scan', scans=None):
    """Return a mock MapConfig for make_pipeline tests.

    scans: list of (spec_file, scan_number, npts, shape) tuples.
    """
    if scans is None:
        scans = [('/spec/scan.spec', 1, 10, [10])]
    mock_spec_scans = []
    for spec_file, scan_number, npts, shape in scans:
        sp = MagicMock()
        sp.spec_scan_npts = npts
        sp.spec_scan_shape = shape
        sg = MagicMock()
        sg.spec_file = spec_file
        sg.scan_numbers = [scan_number]
        sg.get_scanparser.return_value = sp
        mock_spec_scans.append(sg)
    mc = MagicMock()
    mc.title = title
    mc.spec_scans = mock_spec_scans
    return mc


# ── VerboseSafeDumper ─────────────────────────────────────────────────────────

class TestVerboseSafeDumper:
    def test_shared_reference_produces_no_alias(self):
        shared = {'key': 'value'}
        out = yaml.dump({'a': shared, 'b': shared}, Dumper=VerboseSafeDumper)
        assert '*' not in out
        assert '&' not in out

    def test_output_roundtrips(self):
        data = {'x': [1, 2, 3], 'y': {'nested': True}}
        out = yaml.dump(data, Dumper=VerboseSafeDumper)
        assert yaml.safe_load(out) == data

    def test_nested_shared_reference(self):
        node = [1, 2, 3]
        out = yaml.dump({'p': node, 'q': node, 'r': node}, Dumper=VerboseSafeDumper)
        assert out.count('- 1') == 3


# ── convert_configs ───────────────────────────────────────────────────────────

class TestConvertConfigsOutputFiles:
    def test_all_three_files_created(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        assert (tmp_path / 'detector_config.yaml').exists()
        assert (tmp_path / 'pyfai_integration_processor_config.yaml').exists()
        assert (tmp_path / 'corrections_config.yaml').exists()

    def test_absolute_output_filenames_respected(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        _write_integration(tmp_path / 'tool.yaml')
        det = str(sub / 'det.yaml')
        pyfai = str(sub / 'pyfai.yaml')
        corr = str(sub / 'corr.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')],
                        detector_filename=det,
                        pyfai_filename=pyfai,
                        correction_filename=corr)
        assert Path(det).exists()
        assert Path(pyfai).exists()
        assert Path(corr).exists()

    def test_relative_output_filenames_resolved_against_outputdir(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')],
                        detector_filename='custom_det.yaml')
        assert (tmp_path / 'custom_det.yaml').exists()


class TestConvertConfigsIntegrationMethod:
    @pytest.mark.parametrize('integration_type,expected_method', [
        ('azimuthal', 'integrate1d'),
        ('radial',    'integrate_radial'),
        ('cake',      'integrate2d'),
    ])
    def test_integration_method(self, tmp_path, integration_type, expected_method):
        _write_integration(tmp_path / 'tool.yaml', integration_type=integration_type)
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        assert pyfai['integrations'][0]['integration_method'] == expected_method

    def test_radial_uses_azimuthal_npt_as_npt(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml', integration_type='radial',
                           radial_npt=50, azimuthal_npt=36)
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        params = pyfai['integrations'][0]['integration_params']
        assert params['npt'] == 36
        assert params['npt_rad'] == 50

    def test_azimuthal_uses_radial_npt_as_npt(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml', integration_type='azimuthal',
                           radial_npt=200)
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        assert pyfai['integrations'][0]['integration_params']['npt'] == 200

    def test_cake_uses_both_npt_fields(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml', integration_type='cake',
                           radial_npt=80, azimuthal_npt=45)
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        params = pyfai['integrations'][0]['integration_params']
        assert params['npt_rad'] == 80
        assert params['npt_azim'] == 45

    def test_azimuthal_and_cake_have_multi_geometry(self, tmp_path):
        for itype in ('azimuthal', 'cake'):
            _write_integration(tmp_path / f'{itype}.yaml', integration_type=itype)
        for itype in ('azimuthal', 'cake'):
            convert_configs(str(tmp_path), [str(tmp_path / f'{itype}.yaml')])
            _, pyfai, _ = _read_outputs(tmp_path)
            assert 'multi_geometry' in pyfai['integrations'][0], itype

    def test_radial_does_not_have_multi_geometry(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml', integration_type='radial')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        assert 'multi_geometry' not in pyfai['integrations'][0]

    def test_integration_name_from_title(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml', title='waxs_1d')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        assert pyfai['integrations'][0]['name'] == 'waxs_1d'

    def test_multiple_tools_produce_multiple_integrations(self, tmp_path):
        _write_integration(tmp_path / 'a.yaml', title='a')
        _write_integration(tmp_path / 'b.yaml', title='b')
        convert_configs(str(tmp_path),
                        [str(tmp_path / 'a.yaml'), str(tmp_path / 'b.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        assert len(pyfai['integrations']) == 2


class TestConvertConfigsDetectors:
    @pytest.mark.parametrize('prefix,expected_shape', [
        ('PIL5',  [619, 487]),
        ('PIL9',  [407, 487]),
        ('PIL11', [407, 487]),
    ])
    def test_known_detector_shape(self, tmp_path, prefix, expected_shape):
        dets = [{'prefix': prefix, 'poni_file': '/p.poni', 'mask_file': '/m.tif'}]
        _write_integration(tmp_path / 'tool.yaml', detectors=dets)
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        det_cfg, _, _ = _read_outputs(tmp_path)
        det = next(d for d in det_cfg['detectors'] if d['id'] == prefix)
        assert det['shape'] == expected_shape

    def test_unknown_detector_gets_placeholder_shape(self, tmp_path):
        dets = [{'prefix': 'NEWDET', 'poni_file': '/p.poni', 'mask_file': '/m.tif'}]
        _write_integration(tmp_path / 'tool.yaml', detectors=dets)
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        det_cfg, _, _ = _read_outputs(tmp_path)
        det = next(d for d in det_cfg['detectors'] if d['id'] == 'NEWDET')
        assert det['shape'] == [1, 1]

    def test_detector_deduplicated_across_tools(self, tmp_path):
        dets = [{'prefix': 'PIL5', 'poni_file': '/p.poni', 'mask_file': '/m.tif'}]
        _write_integration(tmp_path / 'a.yaml', title='a', detectors=dets)
        _write_integration(tmp_path / 'b.yaml', title='b', detectors=dets)
        convert_configs(str(tmp_path),
                        [str(tmp_path / 'a.yaml'), str(tmp_path / 'b.yaml')])
        det_cfg, _, _ = _read_outputs(tmp_path)
        assert [d['id'] for d in det_cfg['detectors']].count('PIL5') == 1

    def test_azimuthal_integrator_deduplicated_across_tools(self, tmp_path):
        dets = [{'prefix': 'PIL5', 'poni_file': '/p.poni', 'mask_file': '/m.tif'}]
        _write_integration(tmp_path / 'a.yaml', title='a', detectors=dets)
        _write_integration(tmp_path / 'b.yaml', title='b', detectors=dets)
        convert_configs(str(tmp_path),
                        [str(tmp_path / 'a.yaml'), str(tmp_path / 'b.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        ids = [ai['id'] for ai in pyfai['azimuthal_integrators']]
        assert ids.count('PIL5') == 1

    def test_azimuthal_integrator_poni_and_mask_stored(self, tmp_path):
        dets = [{'prefix': 'PIL5', 'poni_file': '/exp/det.poni',
                 'mask_file': '/exp/mask.tif'}]
        _write_integration(tmp_path / 'tool.yaml', detectors=dets)
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, pyfai, _ = _read_outputs(tmp_path)
        ai = pyfai['azimuthal_integrators'][0]
        assert ai['poni_file'] == '/exp/det.poni'
        assert ai['mask_file'] == '/exp/mask.tif'


class TestConvertConfigsCorrections:
    def test_corrections_tool_written_to_corrections_config(self, tmp_path):
        _write_corrections(tmp_path / 'tool.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, _, corr = _read_outputs(tmp_path)
        assert len(corr['corrections']) == 1

    def test_corrections_excludes_tool_type_key(self, tmp_path):
        _write_corrections(tmp_path / 'tool.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, _, corr = _read_outputs(tmp_path)
        assert 'tool_type' not in corr['corrections'][0]

    def test_corrections_excludes_validate_data_present(self, tmp_path):
        _write_corrections(tmp_path / 'tool.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, _, corr = _read_outputs(tmp_path)
        assert 'validate_data_present' not in corr['corrections'][0]

    def test_corrections_preserves_other_fields(self, tmp_path):
        _write_corrections(tmp_path / 'tool.yaml', custom_field='hello')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, _, corr = _read_outputs(tmp_path)
        assert corr['corrections'][0].get('custom_field') == 'hello'

    def test_integration_tool_does_not_appear_in_corrections(self, tmp_path):
        _write_integration(tmp_path / 'tool.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        _, _, corr = _read_outputs(tmp_path)
        assert corr['corrections'] == []

    def test_corrections_tool_does_not_appear_in_detectors(self, tmp_path):
        _write_corrections(tmp_path / 'tool.yaml')
        convert_configs(str(tmp_path), [str(tmp_path / 'tool.yaml')])
        det, _, _ = _read_outputs(tmp_path)
        assert det['detectors'] == []


# ── make_pipeline ─────────────────────────────────────────────────────────────

class TestMakePipeline:
    def _run(self, tmp_path, mock_map, **kwargs):
        with patch('CHAP.common.reader.YAMLReader') as MockReader:
            MockReader.run.return_value = mock_map
            result = make_pipeline(str(tmp_path), **kwargs)
        with open(tmp_path / kwargs.get('pipeline_filename', 'pipeline.yaml')) as f:
            written = yaml.safe_load(f)
        return result, written

    def test_pipeline_file_written(self, tmp_path):
        _, _ = self._run(tmp_path, _make_mock_map())
        assert (tmp_path / 'pipeline.yaml').exists()

    def test_custom_pipeline_filename(self, tmp_path):
        self._run(tmp_path, _make_mock_map(), pipeline_filename='custom.yaml')
        assert (tmp_path / 'custom.yaml').exists()

    def test_returns_dict_with_required_keys(self, tmp_path):
        result, _ = self._run(tmp_path, _make_mock_map())
        assert isinstance(result, dict)
        assert 'config' in result
        assert 'setup' in result
        assert 'convert' in result

    def test_config_root_is_resolved_outputdir(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        assert written['config']['root'] == str(tmp_path.resolve())

    def test_setup_has_five_yaml_readers(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        readers = [s for s in written['setup'] if 'common.reader.YAMLReader' in s]
        assert len(readers) == 5

    def test_setup_yaml_reader_schemas(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        schemas = [s['common.reader.YAMLReader']['schema']
                   for s in written['setup'] if 'common.reader.YAMLReader' in s]
        assert 'common.models.map.DetectorConfig' in schemas
        assert 'common.models.map.MapConfig' in schemas
        assert 'common.models.integration.PyfaiIntegrationConfig' in schemas
        assert 'saxswaxs.models.CorrectionsConfig' in schemas
        assert 'saxswaxs.models.FitsConfig' in schemas

    def test_setup_has_setup_processor(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        procs = [s for s in written['setup']
                 if 'saxswaxs.processor.SetupProcessor' in s]
        assert len(procs) == 1

    def test_setup_processor_raw_data_false(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        proc = next(s for s in written['setup']
                    if 'saxswaxs.processor.SetupProcessor' in s)
        assert proc['saxswaxs.processor.SetupProcessor']['raw_data'] is False

    def test_setup_has_zarr_writer(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        writers = [s for s in written['setup'] if 'common.writer.ZarrWriter' in s]
        assert len(writers) == 1

    def test_zarr_filename_derived_from_map_title(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(title='my_exp'))
        writer = next(s for s in written['setup'] if 'common.writer.ZarrWriter' in s)
        assert writer['common.writer.ZarrWriter']['filename'] == 'my_exp.zarr'

    @pytest.mark.parametrize('scans,expected_update_count', [
        ([('/spec', 1, 10, [10])],         1),   # 1D scan → 1 update
        ([('/spec', 1, 20, [5, 4])],        4),   # 2D scan, 4 rows → 4 updates
        ([('/spec', 1, 10, [10]),
          ('/spec', 2,  5, [5])],           2),   # two 1D scans → 2 updates
    ])
    def test_update_pipeline_count(self, tmp_path, scans, expected_update_count):
        _, written = self._run(tmp_path, _make_mock_map(scans=scans))
        update_keys = [k for k in written if k.startswith('update_')]
        assert len(update_keys) == expected_update_count

    def test_update_pipeline_idx_slice_1d(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(
            scans=[('/spec', 1, 10, [10])]))
        proc = next(s for s in written['update_0']
                    if 'saxswaxs.processor.UpdateValuesProcessor' in s)
        assert proc['saxswaxs.processor.UpdateValuesProcessor']['idx_slice'] == {
            'start': 0, 'stop': 10, 'step': 1,
        }

    def test_update_pipeline_idx_slice_2d_rows(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(
            scans=[('/spec', 1, 20, [5, 4])]))
        for row in range(4):
            proc = next(s for s in written[f'update_{row}']
                        if 'saxswaxs.processor.UpdateValuesProcessor' in s)
            sl = proc['saxswaxs.processor.UpdateValuesProcessor']['idx_slice']
            assert sl == {'start': row * 5, 'stop': row * 5 + 5, 'step': 1}

    def test_update_npts_accumulates_across_scans(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(scans=[
            ('/spec', 1, 10, [10]),
            ('/spec', 2,  5,  [5]),
        ]))
        proc = next(s for s in written['update_1']
                    if 'saxswaxs.processor.UpdateValuesProcessor' in s)
        sl = proc['saxswaxs.processor.UpdateValuesProcessor']['idx_slice']
        assert sl['start'] == 10
        assert sl['stop'] == 15

    def test_dataset_chunks_equals_row_npts_1d(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(
            scans=[('/spec', 1, 10, [10])]))
        proc = next(s for s in written['setup']
                    if 'saxswaxs.processor.SetupProcessor' in s)
        assert proc['saxswaxs.processor.SetupProcessor']['dataset_chunks'] == [10]

    def test_dataset_chunks_equals_row_npts_2d(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(
            scans=[('/spec', 1, 30, [10, 3])]))
        proc = next(s for s in written['setup']
                    if 'saxswaxs.processor.SetupProcessor' in s)
        assert proc['saxswaxs.processor.SetupProcessor']['dataset_chunks'] == [10]

    def test_convert_pipeline_zarr_and_nxs(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(title='exp'))
        assert len(written['convert']) == 1
        cfg = written['convert'][0]['common.processor.ZarrToNexusProcessor']
        assert cfg['zarr_filename'] == 'exp.zarr'
        assert cfg['nexus_filename'] == 'exp.nxs'

    def test_update_pipeline_has_zarr_values_writer(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        writers = [s for s in written['update_0']
                   if 'common.ZarrValuesWriter' in s]
        assert len(writers) == 1

    def test_update_pipeline_zarr_values_writer_idx_slice_matches_processor(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map(
            scans=[('/spec', 1, 10, [10])]))
        proc_sl = next(s for s in written['update_0']
                       if 'saxswaxs.processor.UpdateValuesProcessor' in s
                       )['saxswaxs.processor.UpdateValuesProcessor']['idx_slice']
        writer_sl = next(s for s in written['update_0']
                         if 'common.ZarrValuesWriter' in s
                         )['common.ZarrValuesWriter']['idx_slice']
        assert proc_sl == writer_sl

    def test_reader_filenames_resolved_against_outputdir(self, tmp_path):
        _, written = self._run(tmp_path, _make_mock_map())
        reader = written['setup'][0]['common.reader.YAMLReader']
        assert reader['filename'].startswith(str(tmp_path.resolve()))


# ── saxswaxs_to_chap ──────────────────────────────────────────────────────────

class TestSaxswaxsToChap:
    def test_calls_convert_configs_with_outputdir_and_tools(self, tmp_path):
        tool = str(tmp_path / 'tool.yaml')
        with patch('CHAP.saxswaxs.server.saxswaxs_to_chap.convert_configs') as mock_cc, \
             patch('CHAP.saxswaxs.server.saxswaxs_to_chap.make_pipeline'):
            saxswaxs_to_chap('map.yaml', [tool], str(tmp_path))
            mock_cc.assert_called_once()
            args, kwargs = mock_cc.call_args
            assert args[0] == str(tmp_path)
            assert args[1] == [tool]

    def test_calls_make_pipeline_with_outputdir_and_map(self, tmp_path):
        tool = str(tmp_path / 'tool.yaml')
        with patch('CHAP.saxswaxs.server.saxswaxs_to_chap.convert_configs'), \
             patch('CHAP.saxswaxs.server.saxswaxs_to_chap.make_pipeline') as mock_mp:
            saxswaxs_to_chap('map.yaml', [tool], str(tmp_path))
            mock_mp.assert_called_once()
            args, kwargs = mock_mp.call_args
            assert args[0] == str(tmp_path)
            assert kwargs['map_filename'] == 'map.yaml'

    def test_default_filenames_forwarded_to_convert_configs(self, tmp_path):
        tool = str(tmp_path / 'tool.yaml')
        with patch('CHAP.saxswaxs.server.saxswaxs_to_chap.convert_configs') as mock_cc, \
             patch('CHAP.saxswaxs.server.saxswaxs_to_chap.make_pipeline'):
            saxswaxs_to_chap('map.yaml', [tool], str(tmp_path))
            _, kwargs = mock_cc.call_args
            assert kwargs['detector_filename'] == 'detector_config.yaml'
            assert kwargs['pyfai_filename'] == 'pyfai_integration_processor_config.yaml'
            assert kwargs['correction_filename'] == 'corrections_config.yaml'

    def test_default_filenames_forwarded_to_make_pipeline(self, tmp_path):
        tool = str(tmp_path / 'tool.yaml')
        with patch('CHAP.saxswaxs.server.saxswaxs_to_chap.convert_configs'), \
             patch('CHAP.saxswaxs.server.saxswaxs_to_chap.make_pipeline') as mock_mp:
            saxswaxs_to_chap('map.yaml', [tool], str(tmp_path))
            _, kwargs = mock_mp.call_args
            assert kwargs['fits_filename'] == 'fits_config.yaml'
            assert kwargs['pipeline_filename'] == 'pipeline.yaml'

    def test_custom_filenames_forwarded(self, tmp_path):
        tool = str(tmp_path / 'tool.yaml')
        with patch('CHAP.saxswaxs.server.saxswaxs_to_chap.convert_configs') as mock_cc, \
             patch('CHAP.saxswaxs.server.saxswaxs_to_chap.make_pipeline') as mock_mp:
            saxswaxs_to_chap(
                'map.yaml', [tool], str(tmp_path),
                detector_filename='det.yaml',
                pyfai_filename='pyfai.yaml',
                correction_filename='corr.yaml',
                fits_filename='fits.yaml',
                pipeline_filename='pipe.yaml',
            )
            _, cc_kwargs = mock_cc.call_args
            assert cc_kwargs['detector_filename'] == 'det.yaml'
            assert cc_kwargs['pyfai_filename'] == 'pyfai.yaml'
            assert cc_kwargs['correction_filename'] == 'corr.yaml'
            _, mp_kwargs = mock_mp.call_args
            assert mp_kwargs['detector_filename'] == 'det.yaml'
            assert mp_kwargs['fits_filename'] == 'fits.yaml'
            assert mp_kwargs['pipeline_filename'] == 'pipe.yaml'

    def test_convert_configs_called_before_make_pipeline(self, tmp_path):
        tool = str(tmp_path / 'tool.yaml')
        call_order = []
        with patch('CHAP.saxswaxs.server.saxswaxs_to_chap.convert_configs',
                   side_effect=lambda *a, **kw: call_order.append('cc')), \
             patch('CHAP.saxswaxs.server.saxswaxs_to_chap.make_pipeline',
                   side_effect=lambda *a, **kw: call_order.append('mp')):
            saxswaxs_to_chap('map.yaml', [tool], str(tmp_path))
        assert call_order == ['cc', 'mp']
