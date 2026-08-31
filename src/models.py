from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
import llm_sdk


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


    def get_allowed_token_ids(self, vocab_token_ids: list[int], model: llm_sdk.Small_LLM_Model) -> list[int]:

        allowed_token_ids: list[int] = []

        for token_id in vocab_token_ids:
            decoded = model.decode([token_id])
            if decoded and self.can_append_to_function_name(decoded):
                allowed_token_ids.append(token_id)

        return allowed_token_ids



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


    def append_to_function_name(self, fragment: str) -> None:
        if not self.can_append_to_function_name(fragment):
            raise ValueError(f"Invalid function name fragment: {fragment}")

        self.current_function_name += fragment



class Automate:
    def __init__(self, model: llm_sdk.Small_LLM_Model, vocab_token_ids: list[int], function_defs: list[FunctionDefinition]):

        self.model: llm_sdk.Small_LLM_Model = model
        self.vocab_token_ids: list[int] = vocab_token_ids

        self.sequence: list[str] = [
            "{",
            '"name":',
            "function_name",
            "}"
        ]

        self.sequence_idx = 0
        self.current_sequence = self.sequence[self.sequence_idx]
        self.current_generated_sequence = ""

        self.fonction_name_state: GenerationState = GenerationState(function_defs)


    def increase_sequence(self):

        # Increase sequence if we got a valid function name
        if self.current_sequence == 'function_name':
            if self.fonction_name_state.matches_function_name(self.fonction_name_state.current_function_name):
                self.sequence_idx += 1
                self.current_sequence = self.sequence[self.sequence_idx]
                self.current_generated_sequence = ""
            return

        # Increase sequence for sequence that are related to JSON struct
        if self.current_generated_sequence == self.current_sequence:
            self.sequence_idx += 1
            self.current_sequence = self.sequence[self.sequence_idx]
            self.current_generated_sequence = ""


    def can_append_to_sequence(self, fragment: str) -> bool:
        candidate = self.current_generated_sequence + fragment

        if self.current_sequence.startswith(candidate):
            return True

        return False


    def append_to_sequence(self, fragment: str) -> None:

        if self.current_sequence == 'function_name':
            self.fonction_name_state.append_to_function_name(fragment)
            return

        if not self.can_append_to_sequence(fragment):
            raise ValueError(f"Invalid generated fragment: '{fragment}' for seuqence {self.current_sequence}")

        self.current_generated_sequence += fragment


    def get_current_sequence_allowed_tokens(self) -> list[int]:

        allowed_token_ids = []

        if self.current_sequence == "function_name":
            return self.fonction_name_state.get_allowed_token_ids(self.vocab_token_ids, self.model)


        # Get allowed ids for JSON struct like {,",name:, ect..
        for token_id in self.vocab_token_ids:
            decoded = self.model.decode([token_id])
            if decoded and self.can_append_to_sequence(decoded):
                allowed_token_ids.append(token_id)

        return allowed_token_ids
