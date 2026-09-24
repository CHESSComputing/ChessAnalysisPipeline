"""Unit tests for Pydantic config models in CHAP.saxswaxs.server.chap."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from CHAP.saxswaxs.server.chap import ConvertCfg, SetupCfg, UpdateCfg


def _load(fixture_dir, name):
    return json.loads((fixture_dir / name).read_text())


class TestSetupCfg:
    def test_parses_fixture(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "setup.json")
        data["outputdir"] = str(tmp_path)
        data["data_zarr"] = str(tmp_path / "data.zarr")
        cfg = SetupCfg(**data)
        assert cfg.scan_number == data["scan_number"]
        assert cfg.dwell_time_actual_counter_name == data["dwell_time_actual_counter_name"]
        assert cfg.presample_intensity_counter_name == data["presample_intensity_counter_name"]
        assert isinstance(cfg.spec_file, Path)
        assert isinstance(cfg.detectors_yaml, Path)
        assert isinstance(cfg.map_yaml, Path)
        assert isinstance(cfg.pyfai_yaml, Path)
        assert isinstance(cfg.corrections_yaml, Path)
        assert isinstance(cfg.fits_yaml, Path)
        assert isinstance(cfg.data_zarr, Path)
        assert isinstance(cfg.outputdir, Path)
        assert len(cfg.tool_yamls) == len(data["tool_yamls"])
        assert all(isinstance(p, Path) for p in cfg.tool_yamls)
        assert cfg.dataset_chunks == data["dataset_chunks"]

    def test_missing_required_field_raises(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "setup.json")
        data["outputdir"] = str(tmp_path)
        data["data_zarr"] = str(tmp_path / "data.zarr")
        del data["spec_file"]
        with pytest.raises(ValidationError):
            SetupCfg(**data)

    def test_postsample_intensity_optional(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "setup.json")
        data["outputdir"] = str(tmp_path)
        data["data_zarr"] = str(tmp_path / "data.zarr")
        data.pop("postsample_intensity_counter_name", None)
        cfg = SetupCfg(**data)
        assert cfg.postsample_intensity_counter_name is None

    def test_fits_yaml_present(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "setup.json")
        data["outputdir"] = str(tmp_path)
        data["data_zarr"] = str(tmp_path / "data.zarr")
        cfg = SetupCfg(**data)
        assert cfg.fits_yaml is not None
        assert isinstance(cfg.fits_yaml, Path)


class TestUpdateCfg:
    def test_parses_fixture(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "update.json")
        data["data_zarr"] = str(tmp_path / "data.zarr")
        cfg = UpdateCfg(**data)
        assert cfg.scan_number == data["scan_number"]
        assert isinstance(cfg.spec_file, Path)
        assert isinstance(cfg.data_zarr, Path)
        if "idx_slice_start" in data:
            assert cfg.idx_slice_start == data["idx_slice_start"]
        if "idx_slice_stop" in data:
            assert cfg.idx_slice_stop == data["idx_slice_stop"]
        if "idx_slice_step" in data:
            assert cfg.idx_slice_step == data["idx_slice_step"]

    def test_slice_defaults(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "update.json")
        data["data_zarr"] = str(tmp_path / "data.zarr")
        data.pop("idx_slice_start", None)
        data.pop("idx_slice_stop", None)
        data.pop("idx_slice_step", None)
        cfg = UpdateCfg(**data)
        assert cfg.idx_slice_start == 0
        assert cfg.idx_slice_stop == -1
        assert cfg.idx_slice_step == 1

    def test_missing_required_field_raises(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "update.json")
        data["data_zarr"] = str(tmp_path / "data.zarr")
        del data["scan_number"]
        with pytest.raises(ValidationError):
            UpdateCfg(**data)


class TestConvertCfg:
    def test_parses_fixture(self, fixture_dir, tmp_path):
        data = _load(fixture_dir, "convert.json")
        data["outputdir"] = str(tmp_path)
        cfg = ConvertCfg(**data)
        assert isinstance(cfg.outputdir, Path)

    def test_missing_outputdir_raises(self):
        with pytest.raises(ValidationError):
            ConvertCfg()
