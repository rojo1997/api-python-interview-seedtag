from pydantic import BaseModel, Field, validator
from typing import List
import enum

from exception import ModelSelectionException

N_COLUMNS = 4

def validate_crystal_data_utils(crystalData: List[List[float]]) -> List[List[float]]:
    for row in crystalData:
        if len(row) != N_COLUMNS:
            raise ValueError("Invalid crystalData")
    return crystalData

class Model(str, enum.Enum):
    pytorch = 'pytorch'
    sklearn = 'sklearn'

class Label(str, enum.Enum):
    blue = 'blue'
    green = 'green'
    yellow = 'yellow'

class GonkInput(BaseModel):
    crystalData: List[List[float]] = Field()

    @validator('crystalData')
    def validate_crystal_data(cls, v: List[List[float]]):
        return validate_crystal_data_utils(v)

class AstromechsInput(BaseModel):
    crystalData: List[List[float]] = Field()
    model: str = Field()

    @validator('crystalData')
    def validate_crystal_data(cls, v: List[List[float]]):
        return validate_crystal_data_utils(v)
    
    @validator('model')
    def validate_model(cls, v: str):
        if v not in [Model.pytorch, Model.sklearn]:
            raise ModelSelectionException()

class ScoreRow(BaseModel):
    blue: float = Field(ge = 0.0, le = 1.0)
    green: float = Field(ge = 0.0, le = 1.0)
    yellow: float = Field(ge = 0.0, le = 1.0)

class Output(BaseModel):
    prediction: List[Label] = Field()
    scores: List[ScoreRow]