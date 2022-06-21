
from fastapi import APIRouter, Query, Body
from starlette.requests import Request
from typing import Optional, Any

from core.APILayer import APILayer

from schema import AstromechsInput

router = APIRouter(
    prefix = "/astromech",
    tags = ["astromech"]
)

@router.post("", response_model = Any)
async def post_astromech(
    request: Request,
    input: AstromechsInput = Body(..., examples = {
        "Test1": {
            "summary": "Test1",
            "value": {
                "crystalData": [[0.92, 0.12, 0.31, 0.09]],
                "model": "pytorch"
            }
        },
        "Test2": {
            "summary": "Test2",
            "value": {
                "crystalData": [[0.92, 0.12, 0.31, 0.09]],
                "model": "sklearn"
            }
        },
        "Test3 Fail": {
            "summary": "Test3 Fail Dim",
            "value": {
                "crystalData": [[0.92, 0.12, 0.32, 0.31, 0.09]],
                "model": "sklearn"
            }
        },
        "Test4 Fail": {
            "summary": "Test4 Fail Model",
            "value": {
                "crystalData": [[0.92, 0.12, 0.31, 0.09]],
                "model": "tensorflow"
            }
        }
    })
) -> Any:
    myAPILayer: APILayer = request.app.state.APILayer
    result = myAPILayer.astromech(
        input = input
    )
    return result