"""Tests for the Flask HTTP interface in CHAP.saxswaxs.server.server."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _load(fixture_dir, name):
    return json.loads((fixture_dir / name).read_text())


@pytest.fixture
def client():
    with patch("CHAP.saxswaxs.server.server.put"):
        from CHAP.saxswaxs.server.server import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def _setup_body(fixture_dir, tmp_path):
    data = _load(fixture_dir, "setup.json")
    data["outputdir"] = str(tmp_path)
    data["data_zarr"] = str(tmp_path / "data.zarr")
    return data


def _update_body(fixture_dir, tmp_path):
    data = _load(fixture_dir, "update.json")
    data["data_zarr"] = str(tmp_path / "data.zarr")
    return data


def _convert_body(fixture_dir, tmp_path):
    data = _load(fixture_dir, "convert.json")
    data["outputdir"] = str(tmp_path)
    return data


class TestSetupEndpoint:
    def test_returns_202(self, client, fixture_dir, tmp_path):
        resp = client.post("/setup", json=_setup_body(fixture_dir, tmp_path))
        assert resp.status_code == 202

    def test_returns_queued_status(self, client, fixture_dir, tmp_path):
        resp = client.post("/setup", json=_setup_body(fixture_dir, tmp_path))
        assert resp.get_json() == {"status": "queued"}

    def test_put_called_with_setup_function(self, client, fixture_dir, tmp_path):
        with patch("CHAP.saxswaxs.server.server.put") as mock_put:
            resp = client.post("/setup", json=_setup_body(fixture_dir, tmp_path))
            assert resp.status_code == 202
            mock_put.assert_called_once()
            task, args, kwargs = mock_put.call_args[0]
            assert task.__name__ == "setup"
            assert type(args[0]).__name__ == "SetupCfg"
            assert kwargs == {}

    def test_setup_cfg_fields_correct(self, client, fixture_dir, tmp_path):
        with patch("CHAP.saxswaxs.server.server.put") as mock_put:
            body = _setup_body(fixture_dir, tmp_path)
            client.post("/setup", json=body)
            _, args, _ = mock_put.call_args[0]
            cfg = args[0]
            assert cfg.scan_number == body["scan_number"]
            assert cfg.dwell_time_actual_counter_name == body["dwell_time_actual_counter_name"]
            assert cfg.dataset_chunks == body["dataset_chunks"]

    def test_missing_required_field_returns_error(self, client, fixture_dir, tmp_path):
        body = _setup_body(fixture_dir, tmp_path)
        del body["spec_file"]
        resp = client.post("/setup", json=body)
        assert resp.status_code >= 400

    def test_malformed_json_returns_error(self, client):
        resp = client.post("/setup", data="not json", content_type="application/json")
        assert resp.status_code >= 400


class TestUpdateEndpoint:
    def test_returns_202(self, client, fixture_dir, tmp_path):
        resp = client.post("/update", json=_update_body(fixture_dir, tmp_path))
        assert resp.status_code == 202

    def test_returns_queued_status(self, client, fixture_dir, tmp_path):
        resp = client.post("/update", json=_update_body(fixture_dir, tmp_path))
        assert resp.get_json() == {"status": "queued"}

    def test_put_called_with_update_function(self, client, fixture_dir, tmp_path):
        with patch("CHAP.saxswaxs.server.server.put") as mock_put:
            resp = client.post("/update", json=_update_body(fixture_dir, tmp_path))
            assert resp.status_code == 202
            mock_put.assert_called_once()
            task, args, kwargs = mock_put.call_args[0]
            assert task.__name__ == "update"
            assert type(args[0]).__name__ == "UpdateCfg"

    def test_update_cfg_slice_correct(self, client, fixture_dir, tmp_path):
        with patch("CHAP.saxswaxs.server.server.put") as mock_put:
            body = _update_body(fixture_dir, tmp_path)
            client.post("/update", json=body)
            _, args, _ = mock_put.call_args[0]
            cfg = args[0]
            assert cfg.idx_slice_start == body.get("idx_slice_start", 0)
            assert cfg.idx_slice_stop == body.get("idx_slice_stop", -1)
            assert cfg.idx_slice_step == body.get("idx_slice_step", 1)


class TestConvertEndpoint:
    def test_returns_202(self, client, fixture_dir, tmp_path):
        resp = client.post("/convert", json=_convert_body(fixture_dir, tmp_path))
        assert resp.status_code == 202

    def test_returns_queued_status(self, client, fixture_dir, tmp_path):
        resp = client.post("/convert", json=_convert_body(fixture_dir, tmp_path))
        assert resp.get_json() == {"status": "queued"}

    def test_put_called_with_convert_function(self, client, fixture_dir, tmp_path):
        with patch("CHAP.saxswaxs.server.server.put") as mock_put:
            resp = client.post("/convert", json=_convert_body(fixture_dir, tmp_path))
            assert resp.status_code == 202
            task, args, kwargs = mock_put.call_args[0]
            assert task.__name__ == "convert"
            assert type(args[0]).__name__ == "ConvertCfg"


class TestRequestLogging:
    def test_request_completes_without_logging_error(self, client, fixture_dir, tmp_path):
        with patch("CHAP.saxswaxs.server.server.put"):
            resp = client.post("/setup", json=_setup_body(fixture_dir, tmp_path))
        assert resp.status_code == 202
