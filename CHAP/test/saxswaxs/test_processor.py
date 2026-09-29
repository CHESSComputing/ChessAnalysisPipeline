"""pytest-style unittests for CHAP.saxswaxs.processor module."""

# System modules
from copy import deepcopy
import logging
from unittest.mock import MagicMock, patch

# Third party modules
import numpy as np
import pytest

# Local modules
from CHAP.pipeline import PipelineData
from CHAP.saxswaxs.models import (
    FluxCorrectionConfig,
    FluxAbsorptionCorrectionConfig,
    FluxAbsorptionBackgroundCorrectionConfig,
)
from CHAP.saxswaxs.processor import (
    FluxAbsorptionBackgroundCorrectionProcessor,
    FluxAbsorptionCorrectionProcessor,
    FluxCorrectionProcessor,
    PyfaiIntegrationProcessor,
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


def _make_background_dump(fake_spec):
    """Return a model_dump-compatible dict for a Background, bypassing
    SPEC file validation."""
    from CHAP.saxswaxs.models import Background
    mock_scan = MagicMock()
    mock_fs = MagicMock()
    mock_fs.get_scan_by_number.return_value = mock_scan
    with patch('CHAP.common.models.map.FileSpec', MagicMock(return_value=mock_fs)):
        bg = Background(spec_file=fake_spec, scan_numbers=[1])
    return bg.model_dump()


def _make_flux_config(**kwargs):
    defaults = dict(
        correction_type='flux',
        name='flux_corr',
        input_data_name='my_intg',
        presample_intensity_reference_rate=2.0,
    )
    defaults.update(kwargs)
    return FluxCorrectionConfig(**defaults)


def _make_flux_absorption_config(fake_spec, **kwargs):
    bg_dump = _make_background_dump(fake_spec)
    defaults = dict(
        correction_type='flux_absorption',
        name='flux_abs_corr',
        input_data_name='my_intg',
        presample_intensity_reference_rate=2.0,
        background=bg_dump,
    )
    defaults.update(kwargs)
    with patch('CHAP.common.models.map.FileSpec',
               MagicMock(return_value=MagicMock(
                   **{'get_scan_by_number.return_value': MagicMock()}))):
        return FluxAbsorptionCorrectionConfig(**defaults)


def _make_flux_absorption_bg_config(fake_spec, **kwargs):
    bg_dump = _make_background_dump(fake_spec)
    defaults = dict(
        correction_type='flux_absorption_background',
        name='full_corr',
        input_data_name='my_intg',
        presample_intensity_reference_rate=2.0,
        background=bg_dump,
    )
    defaults.update(kwargs)
    with patch('CHAP.common.models.map.FileSpec',
               MagicMock(return_value=MagicMock(
                   **{'get_scan_by_number.return_value': MagicMock()}))):
        return FluxAbsorptionBackgroundCorrectionConfig(**defaults)


# ---------------------------------------------------------------------------
# FluxCorrectionProcessor
# ---------------------------------------------------------------------------

class TestFluxCorrectionProcessor:
    """Tests for FluxCorrectionProcessor.process()."""

    # Common data parameters
    N, Q = 4, 10
    INTENSITY = np.ones((N, Q)) * 3.0
    PRESAMPLE = np.array([1.0, 2.0, 3.0, 4.0])
    REF_RATE = 2.0

    def _make_data(self, intensity=None, presample=None, dwell=None):
        intensity = self.INTENSITY if intensity is None else intensity
        presample = self.PRESAMPLE if presample is None else presample
        items = [
            PipelineData(name='my_intg', data=intensity.copy()),
            PipelineData(name='presample_intensity', data=presample.copy()),
        ]
        if dwell is not None:
            items.append(PipelineData(name='dwell_time_actual', data=dwell))
        return items

    def test_output_shape(self):
        proc = FluxCorrectionProcessor(config=_make_flux_config())
        result = proc.process(self._make_data())
        assert result.shape == self.INTENSITY.shape

    def test_flux_corrected_values_with_preconfigured_rate(self):
        proc = FluxCorrectionProcessor(config=_make_flux_config())
        result = proc.process(self._make_data())
        expected = self.INTENSITY * (self.REF_RATE / self.PRESAMPLE[:, np.newaxis])
        np.testing.assert_allclose(result, expected)

    def test_flux_corrected_values_computed_rate(self):
        """Reference rate is derived from presample / dwell when not set."""
        dwell = np.ones(self.N)
        config = _make_flux_config(presample_intensity_reference_rate=None)
        proc = FluxCorrectionProcessor(config=config)
        data = self._make_data(dwell=dwell)
        result = proc.process(data)
        ref_rate = float(np.nanmean(self.PRESAMPLE / dwell))
        expected = self.INTENSITY * (ref_rate / self.PRESAMPLE[:, np.newaxis])
        np.testing.assert_allclose(result, expected)

    def test_uniform_presample_scales_uniformly(self):
        """With uniform presample intensity every output row is identical."""
        presample = np.full(self.N, 2.0)
        proc = FluxCorrectionProcessor(config=_make_flux_config())
        data = self._make_data(presample=presample)
        _data = deepcopy(data)
        result = proc.process(_data)
        np.testing.assert_allclose(result, data[0]['data'])

    def test_higher_presample_gives_lower_output(self):
        """Doubling presample at one point halves the corrected intensity."""
        presample = np.array([1.0, 2.0, 1.0, 1.0])
        proc = FluxCorrectionProcessor(config=_make_flux_config())
        data1 = self._make_data(presample=np.full(self.N, 1.0))
        data2 = self._make_data(presample=presample)
        result1 = proc.process(data1)
        proc2 = FluxCorrectionProcessor(config=_make_flux_config())
        result2 = proc2.process(data2)
        np.testing.assert_allclose(result2[1], result1[1] / 2.0)

    def test_1d_integration_result(self):
        """Also works when intensity has only one scan dimension."""
        n, q = 5, 20
        intensity = np.ones((n, q)) * 2.0
        presample = np.linspace(1.0, 3.0, n)
        config = _make_flux_config(presample_intensity_reference_rate=1.5)
        proc = FluxCorrectionProcessor(config=config)
        result = proc.process([
            PipelineData(name='my_intg', data=intensity),
            PipelineData(name='presample_intensity', data=presample),
        ])
        expected = intensity * (1.5 / presample[:, np.newaxis])
        np.testing.assert_allclose(result, expected)

    def test_2d_integration_result(self):
        """Also works for 2D integration output (n, q1, q2)."""
        n, q1, q2 = 3, 5, 6
        intensity = np.ones((n, q1, q2)) * 4.0
        presample = np.array([1.0, 2.0, 4.0])
        config = _make_flux_config(presample_intensity_reference_rate=2.0)
        proc = FluxCorrectionProcessor(config=config)
        result = proc.process([
            PipelineData(name='my_intg', data=intensity),
            PipelineData(name='presample_intensity', data=presample),
        ])
        expected = intensity * (2.0 / presample[:, np.newaxis, np.newaxis])
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# FluxAbsorptionCorrectionProcessor
# ---------------------------------------------------------------------------

class TestFluxAbsorptionCorrectionProcessor:
    """Tests for FluxAbsorptionCorrectionProcessor.process()."""

    N, Q = 3, 8
    INTENSITY = np.ones((N, Q)) * 6.0
    PRESAMPLE = np.array([1.0, 2.0, 3.0])
    POSTSAMPLE = np.array([0.5, 1.0, 1.5])   # post/pre ratio = 0.5 always
    BG_PRESAMPLE = np.array([2.0, 2.0])
    BG_POSTSAMPLE = np.array([1.0, 1.0])     # bg ratio = 0.5
    REF_RATE = 2.0

    # tt = (post/pre) / mean(bg_post/bg_pre) = 0.5 / 0.5 = 1.0

    def _make_data(self):
        return [
            PipelineData(name='my_intg', data=self.INTENSITY.copy()),
            PipelineData(name='presample_intensity', data=self.PRESAMPLE.copy()),
            PipelineData(name='postsample_intensity', data=self.POSTSAMPLE.copy()),
            PipelineData(name='background_presample_intensity',
                         data=self.BG_PRESAMPLE.copy()),
            PipelineData(name='background_postsample_intensity',
                         data=self.BG_POSTSAMPLE.copy()),
        ]

    def test_output_shape(self, fake_spec):
        config = _make_flux_absorption_config(fake_spec)
        proc = FluxAbsorptionCorrectionProcessor(config=config)
        result = proc.process(self._make_data())
        assert result.shape == (self.N, self.Q)

    def test_corrected_values(self, fake_spec):
        """When tt=1 the result equals the flux correction only."""
        config = _make_flux_absorption_config(fake_spec)
        proc = FluxAbsorptionCorrectionProcessor(config=config)
        result = proc.process(self._make_data())
        bg_ratio = np.average(self.BG_POSTSAMPLE / self.BG_PRESAMPLE)
        tt = (self.POSTSAMPLE / self.PRESAMPLE) / bg_ratio
        expected = ((1.0 / tt[:, np.newaxis])
                    * self.INTENSITY
                    * (self.REF_RATE / self.PRESAMPLE[:, np.newaxis]))
        np.testing.assert_allclose(result, expected)

    def test_higher_transmission_gives_lower_output(self, fake_spec):
        """Higher transmission (higher tt) reduces the corrected signal."""
        config = _make_flux_absorption_config(fake_spec)
        proc1 = FluxAbsorptionCorrectionProcessor(config=config)
        # Low-absorption data (tt ~ 1): post ~ pre
        data_lowatten = [
            PipelineData(name='my_intg', data=self.INTENSITY.copy()),
            PipelineData(name='presample_intensity', data=self.PRESAMPLE.copy()),
            PipelineData(name='postsample_intensity', data=self.PRESAMPLE.copy()),
            PipelineData(name='background_presample_intensity',
                         data=self.PRESAMPLE.copy()),
            PipelineData(name='background_postsample_intensity',
                         data=self.PRESAMPLE.copy()),
        ]
        result_lowatten = proc1.process(data_lowatten)

        config2 = _make_flux_absorption_config(fake_spec)
        proc2 = FluxAbsorptionCorrectionProcessor(config=config2)
        # Higher-absorption data (tt > 1)
        data_highatten = [
            PipelineData(name='my_intg', data=self.INTENSITY.copy()),
            PipelineData(name='presample_intensity', data=self.PRESAMPLE.copy()),
            PipelineData(name='postsample_intensity',
                         data=self.PRESAMPLE * 2.0),   # post > pre
            PipelineData(name='background_presample_intensity',
                         data=self.PRESAMPLE.copy()),
            PipelineData(name='background_postsample_intensity',
                         data=self.PRESAMPLE.copy()),
        ]
        result_highatten = proc2.process(data_highatten)
        # Higher tt → lower corrected output
        assert np.all(result_highatten < result_lowatten)


# ---------------------------------------------------------------------------
# FluxAbsorptionBackgroundCorrectionProcessor
# ---------------------------------------------------------------------------

class TestFluxAbsorptionBackgroundCorrectionProcessor:
    """Tests for FluxAbsorptionBackgroundCorrectionProcessor.process()."""

    N, Q = 3, 8
    INTENSITY = np.ones((N, Q)) * 4.0
    PRESAMPLE = np.array([1.0, 2.0, 3.0])
    POSTSAMPLE = np.array([0.5, 1.0, 1.5])   # ratio = 0.5 always → tt = 1
    BG_PRESAMPLE = np.array([2.0, 2.0])
    BG_POSTSAMPLE = np.array([1.0, 1.0])     # bg ratio = 0.5 → tt = 1
    BG_INTENSITY = np.ones(Q) * 0.2
    REF_RATE = 2.0

    def _make_data(self, bg_intensity=None):
        bg_intensity = self.BG_INTENSITY if bg_intensity is None else bg_intensity
        return [
            PipelineData(name='my_intg', data=self.INTENSITY.copy()),
            PipelineData(name='presample_intensity', data=self.PRESAMPLE.copy()),
            PipelineData(name='postsample_intensity', data=self.POSTSAMPLE.copy()),
            PipelineData(name='background_presample_intensity',
                         data=self.BG_PRESAMPLE.copy()),
            PipelineData(name='background_postsample_intensity',
                         data=self.BG_POSTSAMPLE.copy()),
            PipelineData(name='background_intensity', data=bg_intensity.copy()),
        ]

    def _expected(self, t=1.0, bg_intensity=None):
        bg_intensity = self.BG_INTENSITY if bg_intensity is None else bg_intensity
        bg_ratio = np.average(self.BG_POSTSAMPLE / self.BG_PRESAMPLE)
        tt = (self.POSTSAMPLE / self.PRESAMPLE) / bg_ratio
        flux_abs = ((1.0 / tt[:, np.newaxis])
                    * self.INTENSITY
                    * (self.REF_RATE / self.PRESAMPLE[:, np.newaxis]))
        bg_term = (np.broadcast_to(bg_intensity, self.INTENSITY.shape)
                   * (self.REF_RATE / np.average(self.BG_PRESAMPLE)))
        return (1.0 / t) * flux_abs - bg_term

    def test_output_shape(self, fake_spec):
        config = _make_flux_absorption_bg_config(fake_spec)
        proc = FluxAbsorptionBackgroundCorrectionProcessor(config=config)
        result = proc.process(self._make_data())
        assert result.shape == (self.N, self.Q)

    def test_no_thickness_normalization(self, fake_spec):
        config = _make_flux_absorption_bg_config(fake_spec)
        proc = FluxAbsorptionBackgroundCorrectionProcessor(config=config)
        result = proc.process(self._make_data())
        np.testing.assert_allclose(result, self._expected(t=1.0), rtol=1e-10)

    def test_with_sample_thickness_cm(self, fake_spec):
        t = 0.5
        config = _make_flux_absorption_bg_config(fake_spec, sample_thickness_cm=t)
        proc = FluxAbsorptionBackgroundCorrectionProcessor(config=config)
        result = proc.process(self._make_data())
        np.testing.assert_allclose(result, self._expected(t=t), rtol=1e-10)

    def test_thickness_scales_output(self, fake_spec):
        """Halving the thickness doubles the output (relative to no-bg-subtract)."""
        config1 = _make_flux_absorption_bg_config(
            fake_spec, sample_thickness_cm=1.0)
        config2 = _make_flux_absorption_bg_config(
            fake_spec, sample_thickness_cm=0.5)
        proc1 = FluxAbsorptionBackgroundCorrectionProcessor(config=config1)
        proc2 = FluxAbsorptionBackgroundCorrectionProcessor(config=config2)
        # Use zero background so the scaling relationship is simple
        data1 = self._make_data(bg_intensity=np.zeros(self.Q))
        data2 = self._make_data(bg_intensity=np.zeros(self.Q))
        result1 = proc1.process(data1)
        result2 = proc2.process(data2)
        np.testing.assert_allclose(result2, result1 * 2.0, rtol=1e-10)

    def test_zero_background_matches_flux_absorption(self, fake_spec):
        """With zero background the result equals the flux-absorption correction."""
        config_bg = _make_flux_absorption_bg_config(fake_spec)
        proc_bg = FluxAbsorptionBackgroundCorrectionProcessor(config=config_bg)
        result_bg = proc_bg.process(
            self._make_data(bg_intensity=np.zeros(self.Q)))

        config_fa = _make_flux_absorption_config(fake_spec)
        proc_fa = FluxAbsorptionCorrectionProcessor(config=config_fa)
        data_fa = [
            PipelineData(name='my_intg', data=self.INTENSITY.copy()),
            PipelineData(name='presample_intensity', data=self.PRESAMPLE.copy()),
            PipelineData(name='postsample_intensity', data=self.POSTSAMPLE.copy()),
            PipelineData(name='background_presample_intensity',
                         data=self.BG_PRESAMPLE.copy()),
            PipelineData(name='background_postsample_intensity',
                         data=self.BG_POSTSAMPLE.copy()),
        ]
        result_fa = proc_fa.process(data_fa)

        np.testing.assert_allclose(result_bg, result_fa, rtol=1e-10)

    def test_nonzero_background_reduces_output(self, fake_spec):
        """Positive background intensity reduces the corrected output."""
        config = _make_flux_absorption_bg_config(fake_spec)
        proc1 = FluxAbsorptionBackgroundCorrectionProcessor(config=config)
        proc2 = FluxAbsorptionBackgroundCorrectionProcessor(
            config=_make_flux_absorption_bg_config(fake_spec))
        result_no_bg = proc1.process(
            self._make_data(bg_intensity=np.zeros(self.Q)))
        result_with_bg = proc2.process(self._make_data())
        assert np.all(result_with_bg < result_no_bg)


# ---------------------------------------------------------------------------
# PyfaiIntegrationProcessor
# ---------------------------------------------------------------------------

class TestPyfaiIntegrationProcessor:
    """Tests for PyfaiIntegrationProcessor.process() using mocked integrations."""

    @pytest.fixture
    def mock_proc(self):
        """Return a PyfaiIntegrationProcessor backed by a mocked config."""
        n_frames, n_q = 5, 100
        det_id = 'det_0'

        mock_ai = MagicMock()
        mock_ai.get_id.return_value = det_id

        mock_intg = MagicMock()
        mock_intg.name = 'my_intg'
        mock_intg.integrate.return_value = {
            'intensities': np.ones((n_frames, n_q))
        }

        mock_config = MagicMock()
        mock_config.azimuthal_integrators = [mock_ai]
        mock_config.integrations = [mock_intg]

        proc = PyfaiIntegrationProcessor.model_construct(config=mock_config)
        proc.logger = logging.getLogger('test_pyfai')
        return proc, det_id, n_frames, n_q

    def test_process_returns_list(self, mock_proc):
        proc, det_id, n_frames, n_q = mock_proc
        det_data = np.zeros((n_frames, 64, 64))
        result = proc.process([PipelineData(name=det_id, data=det_data)])
        assert isinstance(result, list)

    def test_process_result_has_path_and_data(self, mock_proc):
        proc, det_id, n_frames, n_q = mock_proc
        det_data = np.zeros((n_frames, 64, 64))
        result = proc.process([PipelineData(name=det_id, data=det_data)])
        assert len(result) == 1
        assert 'path' in result[0]
        assert 'data' in result[0]

    def test_process_path_includes_integration_name(self, mock_proc):
        proc, det_id, n_frames, n_q = mock_proc
        det_data = np.zeros((n_frames, 64, 64))
        result = proc.process([PipelineData(name=det_id, data=det_data)])
        assert result[0]['path'] == 'my_intg/data/I'

    def test_process_data_shape(self, mock_proc):
        proc, det_id, n_frames, n_q = mock_proc
        det_data = np.zeros((n_frames, 64, 64))
        result = proc.process([PipelineData(name=det_id, data=det_data)])
        assert result[0]['data'].shape == (n_frames, n_q)

    def test_process_calls_integrate_once_per_integration(self, mock_proc):
        proc, det_id, n_frames, n_q = mock_proc
        det_data = np.zeros((n_frames, 64, 64))
        proc.process([PipelineData(name=det_id, data=det_data)])
        proc.config.integrations[0].integrate.assert_called_once()

    def test_process_multiple_integrations(self):
        """Two integration configs produce two result entries."""
        n_frames, n_q = 3, 50
        det_id = 'det_0'

        mock_ai = MagicMock()
        mock_ai.get_id.return_value = det_id

        results_by_name = {
            'intg_a': np.ones((n_frames, n_q)),
            'intg_b': np.ones((n_frames, n_q)) * 2.0,
        }

        integrations = []
        for name, intens in results_by_name.items():
            m = MagicMock()
            m.name = name
            m.integrate.return_value = {'intensities': intens}
            integrations.append(m)

        mock_config = MagicMock()
        mock_config.azimuthal_integrators = [mock_ai]
        mock_config.integrations = integrations

        proc = PyfaiIntegrationProcessor.model_construct(config=mock_config)
        proc.logger = logging.getLogger('test_pyfai_multi')

        det_data = np.zeros((n_frames, 64, 64))
        result = proc.process([PipelineData(name=det_id, data=det_data)])
        assert len(result) == 2
        paths = {r['path'] for r in result}
        assert paths == {'intg_a/data/I', 'intg_b/data/I'}
