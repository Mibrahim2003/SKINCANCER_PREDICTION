"""Simple smoke tests for the FastAPI endpoints."""

import json
import os
import random
import sys
from pathlib import Path

import requests

# Adjust if you run the server on a different host/port
BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
IMAGE_ENDPOINT = f"{BASE_URL}/predict/image"
FEATURE_ENDPOINT = f"{BASE_URL}/predict/features"

# Folder holding sample images; update if your path differs
IMAGES_DIR = Path("HAM10000_images_part_1")


def pick_random_image() -> Path:
    candidates = list(IMAGES_DIR.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No .jpg files found in {IMAGES_DIR.resolve()}")
    return random.choice(candidates)


def test_image_prediction():
    img_path = pick_random_image()
    print(f"[image] Using {img_path}")
    with img_path.open("rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        resp = requests.post(IMAGE_ENDPOINT, files=files, timeout=30)
    print(f"[image] Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


def test_feature_prediction():
    dummy_features = [random.random() for _ in range(36)]
    payload = {"features": dummy_features}
    resp = requests.post(FEATURE_ENDPOINT, json=payload, timeout=30)
    print(f"[features] Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


def main():
    try:
        test_image_prediction()
    except Exception as exc:  # pragma: no cover (smoke script)
        print(f"[image] Error: {exc}", file=sys.stderr)

    try:
        test_feature_prediction()
    except Exception as exc:  # pragma: no cover (smoke script)
        print(f"[features] Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
