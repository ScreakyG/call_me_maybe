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

        if self.matches_function_name(self.current_function_name):
            encoded = model.encode("\"")[0].tolist()
            allowed_token_ids.extend(encoded)

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


    def set_select_function(self, function_name: str) -> None:
        if not self.matches_function_name(function_name):
            raise ValueError(f"No functions found in function_defs for name: {function_name}")

        for function in self.function_defs:
            if function.name == function_name:
                self.selected_function = function


    def append_to_function_name(self, fragment: str) -> None:

        if fragment == '"':
            self.complete = True
            self.set_select_function(self.current_function_name)
            return

        if not self.can_append_to_function_name(fragment):
            raise ValueError(f"Invalid function name fragment: {fragment}")

        self.current_function_name += fragment



class ParametersAutomate:

    def __init__(self, parameters: dict[str, ParameterDefinition]):
        self.complete = False
        self.function_params: dict[str, ParameterDefinition] = parameters

        self.sequence = self.build_params_sequence()

        self.sequence_idx = 0
        self.current_sequence = self.sequence[self.sequence_idx]
        self.current_generated_sequence = ""


    def build_params_sequence(self) -> list[str]:
        params_sequence: list[str] = ['{']

        for name, value in self.function_params.items():
            params_sequence.append(f'"{name}": ')
            params_sequence.append('"')
            params_sequence.append(value.type.name)
            # params_sequence.append('"')

            params_sequence.append(', ')

        params_sequence.append('}')

        return params_sequence

    def stop_sequence(self) -> bool:
        if self.sequence_idx == len(self.sequence):
            return True

        return False


    def increase_sequence(self) -> None:

        # if self.current_sequence == "NUMBER":
        #     self.sequence_idx += 1
        #     if not self.stop_sequence():
        #         self.current_sequence = self.sequence[self.sequence_idx]
        #         self.current_generated_sequence = ""
        #     else:
        #         self.complete = True

        # Increase sequence for sequence that are related to JSON struct
        if self.current_generated_sequence == self.current_sequence:
            self.sequence_idx += 1
            if not self.stop_sequence():
                self.current_sequence = self.sequence[self.sequence_idx]
                self.current_generated_sequence = ""
            else:
                self.complete = True


    def can_append_to_sequence(self, fragment: str) -> bool:
        candidate = self.current_generated_sequence + fragment

        if self.current_sequence == 'STRING':
            return True

        # Only numbers token are allowed in a NUMBER sequence (this may need some tweaking)
        if self.current_sequence == 'NUMBER':
            if fragment in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '"', '-']:
                return True
            return False


        if self.current_sequence.startswith(candidate):
            return True

        return False


    def end_param_value_sequence(self) -> None:
        self.sequence_idx += 1

        if not self.stop_sequence():
            self.current_sequence = self.sequence[self.sequence_idx]
            self.current_generated_sequence = ""
        else:
            self.complete = True


    def append_to_sequence(self, fragment: str) -> None:

        if not self.can_append_to_sequence(fragment):
            raise ValueError(f"Invalid generated fragment: '{fragment}' for sequence {self.current_sequence}")

        # If the LLM choose '"' in a NUMBER sequence, it means it think the number is finished and we can go the next sequence
        if self.current_sequence == 'NUMBER' and fragment == '"':
            self.end_param_value_sequence()
            return

        # This is not working because model can produce a token that do not exactly match '"' when decoded
        if self.current_sequence == 'STRING' and fragment == '"':
            self.end_param_value_sequence()
            return

        self.current_generated_sequence += fragment



    def get_allowed_token_ids(self, vocab_token_ids: list[int], model: llm_sdk.Small_LLM_Model) -> list[int]:

        allowed_token_ids: list[int] = []

        if self.current_sequence == 'STRING':
            return vocab_token_ids

        if self.current_sequence == 'NUMBER':
            for token_id in vocab_token_ids:
                try:
                    decoded = model.decode([token_id])
                    if decoded and self.can_append_to_sequence(decoded):
                        allowed_token_ids.append(token_id)
                except ValueError:
                    pass

        else:
            for token_id in vocab_token_ids:
                decoded = model.decode([token_id])
                if decoded and self.can_append_to_sequence(decoded):
                    allowed_token_ids.append(token_id)

        return allowed_token_ids


class Automate:
    def __init__(self, model: llm_sdk.Small_LLM_Model, prompt: str, vocab_token_ids: list[int], function_defs: list[FunctionDefinition]):

        self.model: llm_sdk.Small_LLM_Model = model
        self.prompt: str = prompt
        self.vocab_token_ids: list[int] = vocab_token_ids

        self.sequence: list[str] = [
            "{",
            '"prompt": "',
            'prompt_input',
            '", '
            '"name": "',
            "function_name",
            ", ",
            '"parameters": ',
            "function_params",
            "}"
        ]

        self.sequence_idx = 0
        self.current_sequence = self.sequence[self.sequence_idx]
        self.current_generated_sequence = ""

        # Function name and function parameters handlers
        # They are used when the sequence is 'function_name' or 'function_params'
        self.fonction_name_state: GenerationState = GenerationState(function_defs)
        self.function_params_state: ParametersAutomate | None = None


    def stop_sequence(self) -> bool:
        if self.sequence_idx == len(self.sequence):
            return True

        return False


    def increase_sequence(self) -> None:

        if self.current_sequence == "prompt_input":
            if self.current_generated_sequence == self.prompt:
                self.sequence_idx += 1
                if not self.stop_sequence():
                    self.current_sequence = self.sequence[self.sequence_idx]
                    self.current_generated_sequence = ""
            return

        # Increase sequence if we got a valid function name
        if self.current_sequence == 'function_name':
            if self.fonction_name_state.complete:
                self.sequence_idx += 1
                self.function_params_state = ParametersAutomate(self.fonction_name_state.selected_function.parameters)
                if not self.stop_sequence():
                    self.current_sequence = self.sequence[self.sequence_idx]
                    self.current_generated_sequence = ""
            return


        # Increase Automate sequence if ParametersAutomate is complete, else increase ParametersAutomate sequence
        if self.current_sequence == 'function_params':
            if self.function_params_state.complete:
                self.sequence_idx += 1
                if not self.stop_sequence():
                    self.current_sequence = self.sequence[self.sequence_idx]
                    self.current_generated_sequence = ""
            else:
                self.function_params_state.increase_sequence()
                if self.function_params_state.complete:
                    self.sequence_idx += 1
                    if not self.stop_sequence():
                        self.current_sequence = self.sequence[self.sequence_idx]
                        self.current_generated_sequence = ""

            return


        # Increase sequence for sequence that are related to JSON struct
        if self.current_generated_sequence == self.current_sequence:
            self.sequence_idx += 1
            if not self.stop_sequence():
                self.current_sequence = self.sequence[self.sequence_idx]
                self.current_generated_sequence = ""


    def can_append_to_sequence(self, fragment: str) -> bool:
        candidate = self.current_generated_sequence + fragment

        if self.current_sequence.startswith(candidate):
            return True

        return False


    def append_to_sequence(self, fragment: str) -> None:

        if self.current_sequence == "prompt_input":
            self.current_generated_sequence += fragment
            return

        if self.current_sequence == 'function_name':
            self.fonction_name_state.append_to_function_name(fragment)
            return

        if self.current_sequence == 'function_params':
            self.function_params_state.append_to_sequence(fragment)
            return

        if not self.can_append_to_sequence(fragment):
            raise ValueError(f"Invalid generated fragment: '{fragment}' for sequence {self.current_sequence}")

        self.current_generated_sequence += fragment


    def get_current_sequence_allowed_tokens(self) -> list[int]:

        allowed_token_ids = []

        if self.current_sequence == "prompt_input":
            for token_id in self.vocab_token_ids:
                decoded = self.model.decode([token_id])
                if decoded and self.prompt.startswith(self.current_generated_sequence + decoded):
                    allowed_token_ids.append(token_id)


        elif self.current_sequence == "function_name":
            return self.fonction_name_state.get_allowed_token_ids(self.vocab_token_ids, self.model)


        elif self.current_sequence == "function_params":
            return self.function_params_state.get_allowed_token_ids(self.vocab_token_ids, self.model)


        # Get allowed ids for JSON struct like {,",name:, ect..
        else:
            for token_id in self.vocab_token_ids:
                decoded = self.model.decode([token_id])
                if decoded and self.can_append_to_sequence(decoded):
                    allowed_token_ids.append(token_id)

        return allowed_token_ids
