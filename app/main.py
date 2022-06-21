from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from decouple import config

from routers import (
    astromech,
    pytorch,
    sklearn
)

from events.event_handlers import (
    start_app_handler,
    stop_app_handler
)

from exception import ModelSelectionException

app = FastAPI(
    title = 'Seedtag: api-interview',
    description = open("README.md", mode = "r").read()
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True
)

app.include_router(astromech.router)
app.include_router(pytorch.router)
app.include_router(sklearn.router)

app.add_event_handler("startup", start_app_handler(app))
app.add_event_handler("shutdown", stop_app_handler(app))

@app.exception_handler(ModelSelectionException)
async def model_selection_exception_handler(request: Request, exc: ModelSelectionException):
    return JSONResponse(
        status_code = 400,
        content = {"message": "Selected model is not valid"},
    )

if __name__ == "__main__":
    uvicorn.run(
        app = "main:app", 
        host = "0.0.0.0", 
        port = config("FASTAPI_PORT", default = 3000, cast = int), 
        log_level = config("FASTAPI_LOG", default = "debug", cast = str),
        workers = config("FASTAPI_WORKERS", default = 1, cast = int)
    )