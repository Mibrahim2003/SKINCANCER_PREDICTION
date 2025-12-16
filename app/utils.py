"""Utility helpers for inference-time feature extraction and artifact loading."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops

PathOrArray = Union[str, os.PathLike, np.ndarray]


def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """Load the trained model, scaler, encoder, and metadata from disk."""
    model_path = os.path.join(models_dir, "skin_cancer_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    metadata_path = os.path.join(models_dir, "model_metadata.pkl")

    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "encoder": joblib.load(encoder_path),
        "metadata": joblib.load(metadata_path),
    }


def extract_advanced_features(image_input: PathOrArray, img_size: tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
    """Extract color (HSV) and texture (GLCM) features from an image path or array."""
    if isinstance(image_input, (str, os.PathLike)):
        img = cv2.imread(str(image_input))
    else:
        img = image_input

    if img is None:
        return None

    img = cv2.resize(img, img_size)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_img], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_img], [2], None, [8], [0, 256]).flatten()

    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = np.mean(graycoprops(glcm, "contrast"))
    energy = np.mean(graycoprops(glcm, "energy"))
    homogeneity = np.mean(graycoprops(glcm, "homogeneity"))
    correlation = np.mean(graycoprops(glcm, "correlation"))

    features = np.concatenate(
        [
            hist_h,
            hist_s,
            hist_v,
            [contrast, energy, homogeneity, correlation],
        ]
    )

    return features


def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes into an OpenCV BGR image."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def extract_features_from_bytes(image_bytes: bytes, img_size: tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
    """Convenience wrapper to decode bytes then run the feature extractor."""
    img = preprocess_image(image_bytes)
    if img is None:
        return None
    return extract_advanced_features(img, img_size=img_size)


__all__ = [
    "load_artifacts",
    "extract_advanced_features",
    "preprocess_image",
    "extract_features_from_bytes",
]
