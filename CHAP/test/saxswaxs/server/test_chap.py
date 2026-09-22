"""Unit tests for CHAP.saxswaxs.server.chap with all CHAP processors mocked."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Fixtures setup_cfg, update_cfg, convert_cfg come from conftest.py

CHAP_MOCKS = {
    "CHAP.saxswaxs.server.chap.YAMLReader": MagicMock,
    "CHAP.saxswaxs.server.chap.ZarrWriter": MagicMock,
    "CHAP.saxswaxs.server.chap.ZarrValuesWriter": MagicMock,
    "CHAP.saxswaxs.server.chap.SetupProcessor": MagicMock,
    "CHAP.saxswaxs.server.chap.UpdateValuesProcessor": MagicMock,
    "CHAP.saxswaxs.server.chap.PipelineData": MagicMock,
    "CHAP.saxswaxs.server.chap.setup_configs": MagicMock,
    "CHAP.saxswaxs.server.chap.read_configs": MagicMock,
    "CHAP.saxswaxs.server.chap.cache_clear": MagicMock,
}


def _patch_all():
    """Return a single patch.multiple context manager covering all CHAP seams."""
    return patch.multiple("CHAP.saxswaxs.server.chap", **{k.split(".")[-1]: MagicMock() for k in CHAP_MOCKS})


class TestCacheClear:
    def test_cache_clear_called_in_setup(self, setup_cfg):
        with patch("CHAP.saxswaxs.server.chap.cache_clear") as mock_cc, \
             patch("CHAP.saxswaxs.server.chap.setup_configs"), \
             patch("CHAP.saxswaxs.server.chap.read_configs", return_value=[MagicMock()] * 5), \
             patch("CHAP.saxswaxs.server.chap.SetupProcessor"), \
             patch("CHAP.saxswaxs.server.chap.ZarrWriter"), \
             patch("CHAP.saxswaxs.server.chap.PipelineData"):
            from CHAP.saxswaxs.server.chap import setup
            setup(setup_cfg)
            mock_cc.assert_called_once()

    def test_cache_clear_called_in_update(self, update_cfg):
        with patch("CHAP.saxswaxs.server.chap.cache_clear") as mock_cc, \
             patch("CHAP.saxswaxs.server.chap.read_configs", return_value=[MagicMock()] * 5), \
             patch("CHAP.saxswaxs.server.chap.UpdateValuesProcessor"), \
             patch("CHAP.saxswaxs.server.chap.ZarrValuesWriter"), \
             patch("CHAP.saxswaxs.server.chap.PipelineData"):
            from CHAP.saxswaxs.server.chap import update
            update(update_cfg)
            mock_cc.assert_called_once()


class TestSetup:
    def _run(self, cfg):
        mock_data = [MagicMock()] * 5
        mock_zarr_tree = MagicMock()
        mock_pipeline_data = MagicMock(return_value=mock_zarr_tree)
        mock_setup_proc = MagicMock()
        mock_setup_proc.run = MagicMock(return_value=MagicMock())
        mock_zarr_writer = MagicMock()

        with patch("CHAP.saxswaxs.server.chap.cache_clear"), \
             patch("CHAP.saxswaxs.server.chap.setup_configs") as mock_sc, \
             patch("CHAP.saxswaxs.server.chap.read_configs", return_value=mock_data) as mock_rc, \
             patch("CHAP.saxswaxs.server.chap.SetupProcessor", mock_setup_proc), \
             patch("CHAP.saxswaxs.server.chap.ZarrWriter", mock_zarr_writer), \
             patch("CHAP.saxswaxs.server.chap.PipelineData", mock_pipeline_data):
            from CHAP.saxswaxs.server.chap import setup
            setup(cfg)
        return mock_sc, mock_rc, mock_setup_proc, mock_zarr_writer, mock_pipeline_data

    def test_setup_configs_called(self, setup_cfg):
        mock_sc, *_ = self._run(setup_cfg)
        mock_sc.assert_called_once_with(setup_cfg)

    def test_read_configs_called_with_correct_paths(self, setup_cfg):
        _, mock_rc, *_ = self._run(setup_cfg)
        mock_rc.assert_called_once_with(
            setup_cfg.detectors_yaml,
            setup_cfg.map_yaml,
            setup_cfg.pyfai_yaml,
            setup_cfg.corrections_yaml,
            setup_cfg.fits_yaml,
        )

    def test_setup_processor_called_with_dataset_chunks(self, setup_cfg):
        _, _, mock_setup_proc, *_ = self._run(setup_cfg)
        mock_setup_proc.run.assert_called_once()
        _, call_kwargs = mock_setup_proc.run.call_args
        assert call_kwargs["dataset_chunks"] == setup_cfg.dataset_chunks
        assert call_kwargs["raw_data"] is False

    def test_zarr_writer_called_with_correct_args(self, setup_cfg):
        _, _, _, mock_zarr_writer, _ = self._run(setup_cfg)
        mock_zarr_writer.run.assert_called_once()
        _, call_kwargs = mock_zarr_writer.run.call_args
        assert call_kwargs["filename"] == str(setup_cfg.data_zarr)
        assert call_kwargs["force_overwrite"] is True


class TestUpdate:
    def _run(self, cfg):
        mock_data = [MagicMock()] * 5
        mock_update_proc = MagicMock()
        mock_update_proc.run = MagicMock(return_value=MagicMock())
        mock_zarr_values_writer = MagicMock()
        mock_pipeline_data = MagicMock()

        with patch("CHAP.saxswaxs.server.chap.cache_clear"), \
             patch("CHAP.saxswaxs.server.chap.read_configs", return_value=mock_data), \
             patch("CHAP.saxswaxs.server.chap.UpdateValuesProcessor", mock_update_proc), \
             patch("CHAP.saxswaxs.server.chap.ZarrValuesWriter", mock_zarr_values_writer), \
             patch("CHAP.saxswaxs.server.chap.PipelineData", mock_pipeline_data):
            from CHAP.saxswaxs.server.chap import update
            update(cfg)
        return mock_update_proc, mock_zarr_values_writer

    def test_update_processor_called_with_correct_slice(self, update_cfg):
        mock_proc, _ = self._run(update_cfg)
        mock_proc.run.assert_called_once()
        _, call_kwargs = mock_proc.run.call_args
        assert call_kwargs["idx_slice"] == {
            "start": update_cfg.idx_slice_start,
            "stop": update_cfg.idx_slice_stop,
            "step": update_cfg.idx_slice_step,
        }

    def test_update_processor_called_with_spec_and_zarr(self, update_cfg):
        mock_proc, _ = self._run(update_cfg)
        _, call_kwargs = mock_proc.run.call_args
        assert call_kwargs["spec_file"] == update_cfg.spec_file
        assert call_kwargs["scan_number"] == update_cfg.scan_number
        assert call_kwargs["filename"] == str(update_cfg.data_zarr)
        assert call_kwargs["raw_data"] is True

    def test_zarr_values_writer_called_with_correct_args(self, update_cfg):
        _, mock_writer = self._run(update_cfg)
        mock_writer.run.assert_called_once()
        _, call_kwargs = mock_writer.run.call_args
        assert call_kwargs["filename"] == str(update_cfg.data_zarr)
        assert call_kwargs["resize_axis"] == 0
        assert call_kwargs["idx_slice"] == {
            "start": update_cfg.idx_slice_start,
            "stop": update_cfg.idx_slice_stop,
            "step": update_cfg.idx_slice_step,
        }
        assert call_kwargs["force_overwrite"] is True


class TestConvert:
    def _run(self, cfg):
        mock_process = MagicMock()
        mock_popen = MagicMock(return_value=mock_process)

        with patch("CHAP.saxswaxs.server.chap.subprocess.Popen", mock_popen):
            from CHAP.saxswaxs.server.chap import convert
            convert(cfg)
        return mock_popen, mock_process

    def test_popen_called_with_chap_convert(self, convert_cfg):
        mock_popen, _ = self._run(convert_cfg)
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert cmd[0] == "CHAP"
        assert cmd[1] == convert_cfg.outputdir / "pipeline.yaml"
        assert cmd[2] == "-p"
        assert cmd[3] == "convert"

    def test_popen_stdout_is_logfile(self, convert_cfg, tmp_path):
        import subprocess as _subprocess
        mock_process = MagicMock()

        captured = {}

        def fake_popen(cmd, stdout, stderr):
            captured["stdout"] = stdout
            captured["stderr"] = stderr
            return mock_process

        with patch("CHAP.saxswaxs.server.chap.subprocess.Popen", fake_popen):
            from CHAP.saxswaxs.server.chap import convert
            convert(convert_cfg)

        assert hasattr(captured["stdout"], "write"), "stdout should be a file object"
        assert captured["stderr"] == _subprocess.STDOUT

    def test_logfile_path(self, convert_cfg):
        mock_process = MagicMock()
        opened_paths = []

        original_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if mode == "w":
                opened_paths.append(Path(path))
            return original_open(path, mode, *args, **kwargs)

        with patch("builtins.open", fake_open), \
             patch("CHAP.saxswaxs.server.chap.subprocess.Popen", return_value=mock_process):
            from CHAP.saxswaxs.server.chap import convert
            convert(convert_cfg)

        assert any(p == convert_cfg.outputdir / "chap_convert.log" for p in opened_paths)

    def test_process_wait_called(self, convert_cfg):
        _, mock_process = self._run(convert_cfg)
        mock_process.wait.assert_called_once()
