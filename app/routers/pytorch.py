
from fastapi import APIRouter, Query, Body
from starlette.requests import Request
from typing import Optional, Any

from schema import GonkInput, Output

from core.APILayer import APILayer

router = APIRouter(
    prefix = "/pytorch",
    tags = ["pytorch"]
)

@router.post("", response_model = Output)
async def post_pytorch(
    request: Request,
    input: GonkInput = Body(..., examples = {
        "Test1": {
            "summary": "Test1",
            "value": {
                "crystalData": [[0.92, 0.12, 0.31, 0.09]]
            }
        },
        "Test2": {
            "summary": "Test2",
            "value": {
                "crystalData": [
                    [0.92, 0.12, 0.31, 0.09],
                    [0.31, 0.54, 0.78, 0.17],
                    [0.38, 0.04, 0.21, 0.98]
                ]
            }
        }
    })
) -> Any:
    myAPILayer: APILayer = request.app.state.APILayer
    result = myAPILayer.pytorch(
        input = input
    )
    return result