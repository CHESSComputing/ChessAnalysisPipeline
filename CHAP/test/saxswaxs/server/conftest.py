"""Shared fixtures for CHAP.saxswaxs.server tests."""

import json
from pathlib import Path

import pytest

_DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "saxswaxs"

# Table of fixture directories — add a new entry here to add a new test case set.
FIXTURE_DIRS = [
    _DATA_ROOT / "1",
    _DATA_ROOT / "2",
]


@pytest.fixture(params=FIXTURE_DIRS, ids=[d.name for d in FIXTURE_DIRS])
def fixture_dir(request):
    return request.param


def _load(fixture_dir, name):
    return json.loads((fixture_dir / name).read_text())


@pytest.fixture
def setup_cfg(fixture_dir, tmp_path):
    from CHAP.saxswaxs.server.chap import SetupCfg
    data = _load(fixture_dir, "setup.json")
    data["outputdir"] = str(tmp_path)
    data["data_zarr"] = str(tmp_path / "data.zarr")
    return SetupCfg(**data)


@pytest.fixture
def update_cfg(fixture_dir, tmp_path):
    from CHAP.saxswaxs.server.chap import UpdateCfg
    data = _load(fixture_dir, "update.json")
    data["data_zarr"] = str(tmp_path / "data.zarr")
    return UpdateCfg(**data)


@pytest.fixture
def convert_cfg(fixture_dir, tmp_path):
    from CHAP.saxswaxs.server.chap import ConvertCfg
    data = _load(fixture_dir, "convert.json")
    data["outputdir"] = str(tmp_path)
    return ConvertCfg(**data)
