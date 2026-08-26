from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class PromptInput(BaseModel):
    prompt: str = Field(min_length=1)


class ParameterType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOL = "bool"


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ParameterType


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterDefinition] = Field(min_length=1)
    returns: ParameterDefinition


class OutputBoilerPlate(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str
    name: str
    parameters: str
