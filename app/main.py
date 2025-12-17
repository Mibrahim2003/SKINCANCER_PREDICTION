"""FastAPI application entrypoint for skin cancer inference."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conlist

from . import utils


# Globals populated at startup
model: Any = None
scaler: Any = None
encoder: Any = None
metadata: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoder, metadata
    # Initialize logging
    utils.setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Starting application and loading model artifacts...")
    
    artifacts = utils.load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    metadata = artifacts["metadata"]
    logger.info("Model artifacts loaded successfully")
    yield
    logger.info("Shutting down application...")


app = FastAPI(title="Skin Cancer API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class FeaturesInput(BaseModel):
    features: conlist(float, min_length=36, max_length=36) = Field(
        ..., description="Flat list of 36 engineered features"
    )


def _predict_from_features(features: np.ndarray) -> Dict[str, Any]:
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_2d = features.reshape(1, -1)

    features_scaled = scaler.transform(features_2d)

    preds = model.predict(features_scaled)
    class_id = int(preds[0])
    class_code = encoder.inverse_transform([class_id])[0]
    class_name = metadata.get("class_mapping", {}).get(class_code, class_code)

    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)
        confidence = float(np.max(proba[0]))

    return {
        "class_id": class_id,
        "class_code": class_code,
        "class_name": class_name,
        "confidence": confidence,
    }


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Received image prediction request")
    
    try:
        raw_bytes = await file.read()
        features = utils.extract_features_from_bytes(raw_bytes)
        if features is None:
            logger.warning("Invalid or unreadable image received")
            raise HTTPException(status_code=400, detail="Invalid or unreadable image")

        result = _predict_from_features(features)
        logger.info(f"Prediction success: {result['class_name']} with confidence {result['confidence']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.post("/predict/features")
async def predict_features(payload: FeaturesInput) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Received features prediction request")
    
    try:
        features_array = np.array(payload.features, dtype=float)
        result = _predict_from_features(features_array)
        logger.info(f"Prediction success: {result['class_name']} with confidence {result['confidence']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
