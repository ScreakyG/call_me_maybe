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


class GenerationState:
    def __init__(self, function_defs: list[FunctionDefinition]):
        self.function_defs = function_defs
        self.output = ""
        self.current_function_name = ""
        self.selected_function: FunctionDefinition | None = None
        self.complete = False

    def get_compatible_function_names(self, prefix: str) -> list[str]:
        compatible_functions: list[str] = []

        for function in self.function_defs:
            if function.name.startswith(prefix):
                compatible_functions.append(function.name)

        return compatible_functions


    def matches_function_name(self, name: str) -> bool:
        return any(
            function.name == name
            for function in self.function_defs
        )


    def can_append_to_function_name(self, fragment: str) -> bool:
        candidate = self.current_function_name + fragment

        return bool(
            self.get_compatible_function_names(candidate)
        )
