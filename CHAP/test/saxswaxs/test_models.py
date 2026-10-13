"""pytest-style unittests for CHAP.saxswaxs.models module."""

# System modules
from unittest.mock import MagicMock, patch

# Third party modules
import pytest

# Local modules
from CHAP.common.models.common import IndexSliceConfig
from CHAP.saxswaxs.models import (
    Background,
    CorrectionConfig,
    CorrectionsConfig,
    FluxCorrectionConfig,
    FluxAbsorptionCorrectionConfig,
    FluxAbsorptionBackgroundCorrectionConfig,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_spec(tmp_path):
    """Provide a real temporary file and mock FileSpec so SpecScans
    validators accept it without a genuine SPEC file on disk."""
    fake_path = tmp_path / 'fake.spec'
    fake_path.touch()
    mock_scan = MagicMock()
    mock_fs_instance = MagicMock()
    mock_fs_instance.get_scan_by_number.return_value = mock_scan
    mock_filespec = MagicMock(return_value=mock_fs_instance)
    with patch('CHAP.common.models.map.FileSpec', mock_filespec):
        yield str(fake_path)


def _make_background(fake_spec, **kwargs):
    defaults = dict(spec_file=fake_spec, scan_numbers=[1])
    defaults.update(kwargs)
    mock_scan = MagicMock()
    mock_fs_instance = MagicMock()
    mock_fs_instance.get_scan_by_number.return_value = mock_scan
    mock_filespec = MagicMock(return_value=mock_fs_instance)
    with patch('CHAP.common.models.map.FileSpec', mock_filespec):
        return Background(**defaults)


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

class TestBackground:

    def test_default_idx_slice(self, fake_spec):
        bg = _make_background(fake_spec)
        assert bg.idx_slice.start == 0
        assert bg.idx_slice.stop == -1
        assert bg.idx_slice.step == 1

    def test_scan_step_indices_single_value(self, fake_spec):
        bg = _make_background(fake_spec, scan_step_indices=[3])
        assert bg.idx_slice.start == 3
        assert bg.idx_slice.stop == 4
        assert bg.idx_slice.step == 1

    def test_scan_step_indices_uniform_range(self, fake_spec):
        bg = _make_background(fake_spec, scan_step_indices=[0, 2, 4])
        assert bg.idx_slice.start == 0
        assert bg.idx_slice.stop == 6
        assert bg.idx_slice.step == 2

    def test_scan_step_indices_consecutive(self, fake_spec):
        bg = _make_background(fake_spec, scan_step_indices=[1, 2, 3, 4])
        assert bg.idx_slice.start == 1
        assert bg.idx_slice.stop == 5
        assert bg.idx_slice.step == 1

    def test_scan_step_indices_string(self, fake_spec):
        bg = _make_background(fake_spec, scan_step_indices='0-4')
        assert bg.idx_slice.start == 0
        assert bg.idx_slice.stop == 5
        assert bg.idx_slice.step == 1

    def test_both_idx_slice_and_scan_step_indices_raises(self, fake_spec):
        with pytest.raises(Exception, match='idx_slice or scan_step_indices'):
            _make_background(
                fake_spec,
                idx_slice={'start': 0, 'stop': 5, 'step': 1},
                scan_step_indices=[0, 1, 2, 3, 4],
            )

    def test_non_uniform_scan_step_indices_raises(self, fake_spec):
        with pytest.raises(Exception, match='uniformly spaced'):
            _make_background(fake_spec, scan_step_indices=[0, 1, 3])

    def test_zarr_arrays_returns_i_background_key(self):
        bg = Background.model_construct(
            spec_file='/fake/spec.spec',
            scan_numbers=[1],
            idx_slice=IndexSliceConfig(),
        )
        arrays = bg.zarr_arrays((100,))
        assert 'I_background' in arrays

    def test_zarr_arrays_shape_1d(self):
        bg = Background.model_construct(
            spec_file='/fake/spec.spec',
            scan_numbers=[1],
            idx_slice=IndexSliceConfig(),
        )
        arrays = bg.zarr_arrays((200,))
        assert arrays['I_background']['shape'] == (200,)

    def test_zarr_arrays_shape_2d(self):
        bg = Background.model_construct(
            spec_file='/fake/spec.spec',
            scan_numbers=[1],
            idx_slice=IndexSliceConfig(),
        )
        arrays = bg.zarr_arrays((36, 100))
        assert arrays['I_background']['shape'] == (36, 100)

    def test_zarr_arrays_dtype(self):
        bg = Background.model_construct(
            spec_file='/fake/spec.spec',
            scan_numbers=[1],
            idx_slice=IndexSliceConfig(),
        )
        arrays = bg.zarr_arrays((50,))
        assert arrays['I_background']['dtype'] == 'float64'

    def test_zarr_arrays_has_attributes(self):
        bg = Background.model_construct(
            spec_file='/fake/spec.spec',
            scan_numbers=[1],
            idx_slice=IndexSliceConfig(),
        )
        arrays = bg.zarr_arrays((10,))
        assert 'attributes' in arrays['I_background']


# ---------------------------------------------------------------------------
# FluxCorrectionConfig
# ---------------------------------------------------------------------------

class TestFluxCorrectionConfig:

    def _make(self, **kwargs):
        defaults = dict(
            correction_type='flux',
            name='flux_corr',
            input_data_name='my_intg',
        )
        defaults.update(kwargs)
        return FluxCorrectionConfig(**defaults)

    def test_valid(self):
        cfg = self._make()
        assert cfg.correction_type == 'flux'

    def test_name_alias(self):
        cfg = FluxCorrectionConfig(title='by_title', input_data_name='intg')
        assert cfg.name == 'by_title'

    def test_input_data_name_alias(self):
        cfg = FluxCorrectionConfig(name='c', uncorrected_data_title='intg')
        assert cfg.input_data_name == 'intg'

    def test_input_data_names_single_string(self):
        cfg = self._make()
        assert cfg.input_data_names == ['my_intg']

    def test_input_data_names_list(self):
        cfg = self._make(input_data_name=['src_a', 'src_b'])
        assert cfg.input_data_names == ['src_a', 'src_b']

    def test_processor_name(self):
        cfg = self._make()
        assert cfg.processor_name == 'FluxCorrectionProcessor'

    def test_no_background_by_default(self):
        cfg = self._make()
        assert cfg.background is None

    def test_optional_presample_reference_rate(self):
        cfg = self._make(presample_intensity_reference_rate=3.5)
        assert cfg.presample_intensity_reference_rate == 3.5

    def test_zarr_tree_returns_dict(self):
        cfg = self._make()
        tree = cfg.zarr_tree((5,), [5], (100,))
        assert isinstance(tree, dict)

    def test_zarr_tree_has_data_group(self):
        cfg = self._make()
        tree = cfg.zarr_tree((5,), [5], (100,))
        assert 'children' in tree
        assert 'data' in tree['children']

    def test_zarr_tree_i_corrected_shape_1d_scan(self):
        cfg = self._make()
        tree = cfg.zarr_tree((5,), [5], (100,))
        i_corr = tree['children']['data']['children']['I_corrected']
        assert i_corr['shape'] == (5, 100)

    def test_zarr_tree_i_corrected_shape_2d_scan(self):
        cfg = self._make()
        tree = cfg.zarr_tree((3, 4), [3], (50,))
        i_corr = tree['children']['data']['children']['I_corrected']
        assert i_corr['shape'] == (3, 4, 50)

    def test_zarr_tree_multi_source_creates_per_src_arrays(self):
        cfg = self._make(input_data_name=['src_a', 'src_b'])
        input_shape = {'src_a': (80,), 'src_b': (120,)}
        tree = cfg.zarr_tree((3,), [3], input_shape)
        data_children = tree['children']['data']['children']
        assert 'I_corrected_src_a' in data_children
        assert 'I_corrected_src_b' in data_children
        assert 'I_corrected' not in data_children

    def test_zarr_tree_multi_source_shapes(self):
        cfg = self._make(input_data_name=['src_a', 'src_b'])
        input_shape = {'src_a': (80,), 'src_b': (120,)}
        tree = cfg.zarr_tree((3,), [3], input_shape)
        data_children = tree['children']['data']['children']
        assert data_children['I_corrected_src_a']['shape'] == (3, 80)
        assert data_children['I_corrected_src_b']['shape'] == (3, 120)

    def test_zarr_tree_nxlinks_string(self):
        cfg = self._make()
        tree = cfg.zarr_tree((5,), [5], (100,), nxlinks='/foo/bar')
        nxlinks = tree['children']['data']['attributes']['__nxlinks__']
        assert 'bar' in nxlinks
        assert nxlinks['bar'] == '/foo/bar'

    def test_zarr_tree_nxlinks_list(self):
        cfg = self._make()
        tree = cfg.zarr_tree((5,), [5], (100,), nxlinks=['/a/b', '/c/d'])
        nxlinks = tree['children']['data']['attributes']['__nxlinks__']
        assert 'b' in nxlinks
        assert 'd' in nxlinks

    def test_zarr_tree_no_nxlinks(self):
        cfg = self._make()
        tree = cfg.zarr_tree((5,), [5], (100,))
        data_attrs = tree['children']['data']['attributes']
        assert '__nxlinks__' not in data_attrs

    def test_zarr_tree_correction_type_attribute(self):
        cfg = self._make()
        tree = cfg.zarr_tree((5,), [5], (100,))
        assert tree['attributes']['correction_type'] == 'flux'

    def test_processor_class(self):
        from CHAP.saxswaxs.processor import FluxCorrectionProcessor
        cfg = self._make()
        assert cfg.processor_class is FluxCorrectionProcessor


# ---------------------------------------------------------------------------
# FluxAbsorptionCorrectionConfig
# ---------------------------------------------------------------------------

class TestFluxAbsorptionCorrectionConfig:

    def _make(self, fake_spec, **kwargs):
        bg = _make_background(fake_spec)
        defaults = dict(
            correction_type='flux_absorption',
            name='flux_abs_corr',
            input_data_name='my_intg',
            background=bg.model_dump(),
        )
        defaults.update(kwargs)
        mock_scan = MagicMock()
        mock_fs_instance = MagicMock()
        mock_fs_instance.get_scan_by_number.return_value = mock_scan
        mock_filespec = MagicMock(return_value=mock_fs_instance)
        with patch('CHAP.common.models.map.FileSpec', mock_filespec):
            return FluxAbsorptionCorrectionConfig(**defaults)

    def test_correction_type(self, fake_spec):
        cfg = self._make(fake_spec)
        assert cfg.correction_type == 'flux_absorption'

    def test_processor_name(self, fake_spec):
        cfg = self._make(fake_spec)
        assert cfg.processor_name == 'FluxAbsorptionCorrectionProcessor'

    def test_background_stored(self, fake_spec):
        cfg = self._make(fake_spec)
        assert cfg.background is not None

    def test_background_required(self):
        with pytest.raises(Exception):
            FluxAbsorptionCorrectionConfig(
                name='c', input_data_name='i',
            )

    def test_processor_class(self, fake_spec):
        from CHAP.saxswaxs.processor import FluxAbsorptionCorrectionProcessor
        cfg = self._make(fake_spec)
        assert cfg.processor_class is FluxAbsorptionCorrectionProcessor


# ---------------------------------------------------------------------------
# FluxAbsorptionBackgroundCorrectionConfig
# ---------------------------------------------------------------------------

class TestFluxAbsorptionBackgroundCorrectionConfig:

    def _make(self, fake_spec, **kwargs):
        bg = _make_background(fake_spec)
        defaults = dict(
            correction_type='flux_absorption_background',
            name='full_corr',
            input_data_name='my_intg',
            background=bg.model_dump(),
        )
        defaults.update(kwargs)
        mock_scan = MagicMock()
        mock_fs_instance = MagicMock()
        mock_fs_instance.get_scan_by_number.return_value = mock_scan
        mock_filespec = MagicMock(return_value=mock_fs_instance)
        with patch('CHAP.common.models.map.FileSpec', mock_filespec):
            return FluxAbsorptionBackgroundCorrectionConfig(**defaults)

    def test_correction_type(self, fake_spec):
        cfg = self._make(fake_spec)
        assert cfg.correction_type == 'flux_absorption_background'

    def test_processor_name(self, fake_spec):
        cfg = self._make(fake_spec)
        assert cfg.processor_name == 'FluxAbsorptionBackgroundCorrectionProcessor'

    def test_no_thickness_fields_by_default(self, fake_spec):
        cfg = self._make(fake_spec)
        assert cfg.sample_thickness_cm is None
        assert cfg.sample_mu_inv_cm is None

    def test_only_sample_thickness_cm(self, fake_spec):
        cfg = self._make(fake_spec, sample_thickness_cm=0.5)
        assert cfg.sample_thickness_cm == pytest.approx(0.5)
        assert cfg.sample_mu_inv_cm is None

    def test_only_sample_mu_inv_cm(self, fake_spec):
        cfg = self._make(fake_spec, sample_mu_inv_cm=2.0)
        assert cfg.sample_mu_inv_cm == pytest.approx(2.0)
        assert cfg.sample_thickness_cm is None

    def test_both_thickness_and_mu_inv_raises(self, fake_spec):
        with pytest.raises(Exception, match='sample_thickness_cm OR sample_mu_inv_cm'):
            self._make(fake_spec, sample_thickness_cm=0.1, sample_mu_inv_cm=0.5)

    def test_non_positive_thickness_raises(self, fake_spec):
        with pytest.raises(Exception):
            self._make(fake_spec, sample_thickness_cm=-0.1)

    def test_non_positive_mu_inv_raises(self, fake_spec):
        with pytest.raises(Exception):
            self._make(fake_spec, sample_mu_inv_cm=-1.0)

    def test_processor_class(self, fake_spec):
        from CHAP.saxswaxs.processor import (
            FluxAbsorptionBackgroundCorrectionProcessor,
        )
        cfg = self._make(fake_spec)
        assert cfg.processor_class is FluxAbsorptionBackgroundCorrectionProcessor


# ---------------------------------------------------------------------------
# CorrectionsConfig
# ---------------------------------------------------------------------------

class TestCorrectionsConfig:

    def _flux_corr(self, name, input_name):
        return {
            'correction_type': 'flux',
            'name': name,
            'input_data_name': input_name,
        }

    def test_single_correction_zarr_tree_key(self):
        cfg = CorrectionsConfig(corrections=[self._flux_corr('c1', 'i1')])
        tree = cfg.zarr_tree((5,), [5], {'c1': (100,)})
        assert 'c1' in tree['children']

    def test_multiple_corrections_zarr_tree_keys(self):
        cfg = CorrectionsConfig(corrections=[
            self._flux_corr('c1', 'i1'),
            self._flux_corr('c2', 'i2'),
        ])
        tree = cfg.zarr_tree((3,), [3], {'c1': (50,), 'c2': (80,)})
        assert 'c1' in tree['children']
        assert 'c2' in tree['children']

    def test_zarr_tree_per_correction_shapes(self):
        cfg = CorrectionsConfig(corrections=[
            self._flux_corr('c1', 'i1'),
            self._flux_corr('c2', 'i2'),
        ])
        tree = cfg.zarr_tree((4,), [4], {'c1': (60,), 'c2': (90,)})
        assert (tree['children']['c1']['children']['data']
                    ['children']['I_corrected']['shape'] == (4, 60))
        assert (tree['children']['c2']['children']['data']
                    ['children']['I_corrected']['shape'] == (4, 90))

    def test_zarr_tree_with_per_correction_nxlinks(self):
        cfg = CorrectionsConfig(corrections=[
            self._flux_corr('c1', 'i1'),
        ])
        nxlinks = {'c1': '/entry/data/I'}
        tree = cfg.zarr_tree((3,), [3], {'c1': (50,)}, nxlinks=nxlinks)
        data_attrs = tree['children']['c1']['children']['data']['attributes']
        assert '__nxlinks__' in data_attrs

    def test_zarr_tree_with_shared_nxlinks(self):
        cfg = CorrectionsConfig(corrections=[
            self._flux_corr('c1', 'i1'),
            self._flux_corr('c2', 'i2'),
        ])
        tree = cfg.zarr_tree((3,), [3], {'c1': (50,), 'c2': (80,)},
                             nxlinks='/a/b')
        for name in ('c1', 'c2'):
            data_attrs = (tree['children'][name]['children']
                              ['data']['attributes'])
            assert '__nxlinks__' in data_attrs
