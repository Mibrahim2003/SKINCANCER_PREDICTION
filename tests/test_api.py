"""FastAPI endpoint tests for CI-safe execution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Ensure project root is on the import path when running directly from the tests folder
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app.main as app_module


class DummyModel:
    def predict(self, features):
        return [0]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


DUMMY_CLASS_MAPPING = {"mel": "Melanoma"}


class DummyScaler:
    def transform(self, features):
        return features


class DummyEncoder:
    def inverse_transform(self, class_ids):
        return ["mel"]


@pytest.fixture()
def client(monkeypatch):
    def fake_load_artifacts():
        return {
            "model": DummyModel(),
            "scaler": DummyScaler(),
            "encoder": DummyEncoder(),
            "metadata": {"class_mapping": DUMMY_CLASS_MAPPING},
        }

    monkeypatch.setattr(app_module.utils, "load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(app_module, "model", None)
    monkeypatch.setattr(app_module, "scaler", None)
    monkeypatch.setattr(app_module, "encoder", None)
    monkeypatch.setattr(app_module, "metadata", {})

    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "").lower()


def test_feature_endpoint_returns_class_name(client):
    payload = {"features": [0.0] * 36}
    response = client.post("/predict/features", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "class_name" in body


def test_image_endpoint_handles_bytes(client):
    files = {"file": ("dummy.jpg", b"not an image", "image/jpeg")}
    response = client.post("/predict/image", files=files)
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        body = response.json()
        assert "class_name" in body
