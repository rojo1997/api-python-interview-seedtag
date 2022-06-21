from typing import Callable
from fastapi import FastAPI
import os
from decouple import config

from interview_seedtag import (
    PytorchClassifier, 
    SklearnClassifier
)
import model

from core.APILayer import APILayer

def _startup_model(app: FastAPI) -> None:
    app.state.APILayer = APILayer(
        pytorch_classifier = PytorchClassifier(
            path = model.__file__.replace("__init__.py", "") + "pytorch.model"
        ),
        sklearn_classifier = SklearnClassifier(
            path = model.__file__.replace("__init__.py", "") + "sklearn.model"
        )
    )

def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        _startup_model(app)
    return startup

def _shutdown_model(app: FastAPI) -> None:
    app.state.APILayer = None

def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)
    return shutdown