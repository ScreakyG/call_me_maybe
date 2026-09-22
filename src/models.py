from copy import copy
from enum import Enum
import json
import re
from pydantic import BaseModel, ConfigDict, Field
import llm_sdk


JSON_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')


class PromptInput(BaseModel):
    prompt: str = Field(min_length=1)


class ParameterType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOL = "boolean"


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

        for index, (name, value) in enumerate(self.function_params.items()):
            params_sequence.append('"')
            params_sequence.append(f'{name}"')
            params_sequence.append(': ')

            # If parameter is a string type we wrap it with quotes
            if value.type.name == 'STRING':
                params_sequence.append('"')
                params_sequence.append(value.type.name)

                if index == len(self.function_params) - 1:
                    params_sequence.append('"')
                else:
                    params_sequence.append('", ')

            # If parameter is bool or a number we don't wrap it
            else:
                params_sequence.append(value.type.name)

                if index != len(self.function_params) - 1:
                    params_sequence.append(', ')


        params_sequence.append('}')

        return params_sequence

    def stop_sequence(self) -> bool:
        if self.sequence_idx == len(self.sequence):
            return True

        return False


    def get_allowed_token_ids2(
            self,
            vocab_token_ids: list[int],
            model: llm_sdk.Small_LLM_Model
        ) -> list[int]:

        allowed_tokens_ids: list[int] = []

        print("Current param sequence=", self.current_sequence)
        print("Current generated param sequence =", self.current_generated_sequence)


        for token_id in vocab_token_ids:
            decoded = model.decode([token_id])
            if decoded and self.can_append_to_sequence2(decoded):
                allowed_tokens_ids.append(token_id)

        return allowed_tokens_ids


    def split_token(self, fragment: str, separator: str) -> tuple[str, str]:
        before = ''
        after = ''

        if separator in fragment:
            index = fragment.index(separator)
            before = fragment[:index]
            after = fragment[index + 1:] #Exclude the separator

        return before, after


    def is_quote_escaped(self, before: str) -> bool:
        """Check backslash parity before a quote across token boundaries."""
        prefix = self.current_generated_sequence + before
        backslash_count = 0

        for character in reversed(prefix):
            if character != "\\":
                break
            backslash_count += 1

        return backslash_count % 2 == 1


    def can_sequence_consume_fragment(
            self,
            sequence: str,
            sequence_current_gen: str,
            fragment: str
        ) -> bool:

        return sequence.startswith(sequence_current_gen + fragment)


    def can_consume_quote_fragment(self, fragment) -> bool:
        before, after = self.split_token(fragment, '"')

        # If current sequence consume fragment with quote, the next sequence needs to be valid with whats after the quote
        if self.can_sequence_consume_fragment(self.current_sequence, self.current_generated_sequence, before + '"'):
            if self.can_sequence_consume_fragment(self.sequence[self.sequence_idx + 1], "", after):
                return True

        # If current sequence can't consume fragmentg with quote, the next sequence needs to be valid with '"' + after
        if self.can_sequence_consume_fragment(self.current_sequence, self.current_generated_sequence, before):
            if self.can_sequence_consume_fragment(self.sequence[self.sequence_idx + 1], "", '"' + after):
                return True

        return False


    def can_append_to_sequence2(self, fragment: str) -> bool:
        """Validate a whole token without changing the current state."""
        candidate = copy(self)
        return candidate.consume_fragment(fragment)


    def consume_fragment(self, fragment: str) -> bool:
        """Consume characters and advance through each completed state."""
        if not fragment or self.complete:
            return False

        for character in fragment:
            if self.complete or not self.can_append_character(character):
                return False
            self.append_character(character)
            self.increase_sequence()

        return True


    def can_append_character(self, fragment: str) -> bool:

        # Validate JSON escapes, including those started in previous tokens.
        if self.current_sequence == 'STRING':
            # ord() gives char Unicode, those under 0x20 are control chars (tab, newline, etc..)
            if ord(fragment) < 0x20:
                return False

            pending_unicode = re.search(
                r'(\\+)u[0-9a-fA-F]{0,3}$',
                self.current_generated_sequence,
            )
            if pending_unicode and len(pending_unicode.group(1)) % 2 == 1:
                return fragment in '0123456789abcdefABCDEF'

            # Check if we have impair antislashes , if so we expect a char to be escaped
            if self.is_quote_escaped(''):
                return fragment in '"\\/bfnrtu'

            if '"' in fragment:
                before, after = self.split_token(fragment, '"')

                if self.is_quote_escaped(before):
                    return True

                if self.can_sequence_consume_fragment(self.sequence[self.sequence_idx + 1], "", '"' + after):
                    return True

                return False

            return True

        # A separator is allowed only after a complete JSON number.
        if self.current_sequence == 'NUMBER':
            if fragment in (',', '}'):
                return (
                    JSON_NUMBER.fullmatch(self.current_generated_sequence)
                    is not None
                    and self.sequence[self.sequence_idx + 1].startswith(fragment)
                )

            candidate = self.current_generated_sequence + fragment
            # Incomplete valid prefixes (-, 1., 1e, 1e+) become complete
            # by adding one digit.
            return (
                JSON_NUMBER.fullmatch(candidate) is not None
                or JSON_NUMBER.fullmatch(candidate + '0') is not None
            )


        if self.current_sequence == 'BOOL':
            if fragment in (',', '}'):
                return (
                    self.current_generated_sequence in ('true', 'false')
                    and self.sequence[self.sequence_idx + 1].startswith(fragment)
                )

            candidate = self.current_generated_sequence + fragment
            return 'true'.startswith(candidate) or 'false'.startswith(candidate)


        # For schema structure only
        # If a token has a double quote it means it's a terminating token
        # /!\ We need to add a check if its escaped like : \"
        if '"' in fragment:
            # If we are on the last sequence we simply won't allow quote fragments
            if self.sequence_idx == len(self.sequence) - 1:
                return False

            return self.can_consume_quote_fragment(fragment)


        # For structure sequences we just check if it match the schema
        candidate = self.current_generated_sequence + fragment

        if self.current_sequence.startswith(candidate):
            return True

        return False


    def append_to_sequence2(self, fragment: str) -> None:
        """Commit a token only when every character is valid."""
        candidate = copy(self)
        if not candidate.consume_fragment(fragment):
            raise ValueError(f"Invalid parameter fragment: {fragment!r}")

        self.sequence_idx = candidate.sequence_idx
        self.current_sequence = candidate.current_sequence
        self.current_generated_sequence = candidate.current_generated_sequence
        self.complete = candidate.complete


    def append_character(self, fragment: str) -> None:

        if (
            self.current_sequence in ('NUMBER', 'BOOL')
            and fragment in (',', '}')
        ):
            if not self.can_append_character(fragment):
                raise ValueError(f"Invalid value terminator: {fragment!r}")
            self.end_param_value_sequence()
            self.current_generated_sequence = fragment
            return



        if self.current_sequence == 'STRING' and '"' in fragment:
            before, after = self.split_token(fragment, '"')

            if self.is_quote_escaped(before):
                self.current_generated_sequence += fragment
                return

            else:
                if self.can_sequence_consume_fragment(self.sequence[self.sequence_idx + 1], "", '"' + after):
                    self.current_generated_sequence += before
                    self.end_param_value_sequence()
                    self.current_generated_sequence += '"' + after
                    return

            raise Exception("TOKEN QUOTE PAS ECHAPE PLUS PAS VALIDE POUR PROCHAINE SEQUENCE")


        if '"' in fragment and self.can_append_character(fragment):
            before, after = self.split_token(fragment, '"')

            if self.can_sequence_consume_fragment(self.current_sequence, self.current_generated_sequence, before + '"'):
                self.current_generated_sequence += before + '"'
                self.end_param_value_sequence()

                if self.can_sequence_consume_fragment(self.current_sequence, "", after):
                    self.current_generated_sequence += after

                return

            if self.can_sequence_consume_fragment(self.current_sequence, self.current_generated_sequence, before):
                self.current_generated_sequence += before
                self.end_param_value_sequence()

                if self.can_sequence_consume_fragment(self.current_sequence, "", '"' + after):
                    self.current_generated_sequence += '"' + after

                return


        elif self.can_append_character(fragment):
            self.current_generated_sequence += fragment


    def increase_sequence(self) -> None:

        if self.complete or self.current_sequence in ('STRING', 'NUMBER', 'BOOL'):
            return

        # Increase sequence for sequence that are related to JSON struct
        if self.current_generated_sequence == self.current_sequence:
            self.sequence_idx += 1
            if not self.stop_sequence():
                self.current_sequence = self.sequence[self.sequence_idx]
                self.current_generated_sequence = ""
            else:
                self.complete = True


    def end_param_value_sequence(self) -> None:
        self.sequence_idx += 1

        if not self.stop_sequence():
            self.current_sequence = self.sequence[self.sequence_idx]
            self.current_generated_sequence = ""
        else:
            self.complete = True


class Automate:
    def __init__(self, model: llm_sdk.Small_LLM_Model, prompt: str, vocab_token_ids: list[int], function_defs: list[FunctionDefinition]):

        self.model: llm_sdk.Small_LLM_Model = model
        self.prompt: str = prompt
        self.escaped_prompt: str = json.dumps(prompt, ensure_ascii=False)[1:-1]
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
            if self.current_generated_sequence == self.escaped_prompt:
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
            self.function_params_state.append_to_sequence2(fragment)
            return

        if not self.can_append_to_sequence(fragment):
            raise ValueError(f"Invalid generated fragment: '{fragment}' for sequence {self.current_sequence}")

        self.current_generated_sequence += fragment


    def get_current_sequence_allowed_tokens(self) -> list[int]:

        allowed_token_ids = []

        if self.current_sequence == "prompt_input":
            for token_id in self.vocab_token_ids:
                decoded = self.model.decode([token_id])
                if decoded and self.escaped_prompt.startswith(self.current_generated_sequence + decoded):
                    allowed_token_ids.append(token_id)


        elif self.current_sequence == "function_name":
            return self.fonction_name_state.get_allowed_token_ids(self.vocab_token_ids, self.model)


        elif self.current_sequence == "function_params":
            return self.function_params_state.get_allowed_token_ids2(self.vocab_token_ids, self.model)


        # Get allowed ids for JSON struct like {,",name:, ect..
        else:
            for token_id in self.vocab_token_ids:
                decoded = self.model.decode([token_id])
                if decoded and self.can_append_to_sequence(decoded):
                    allowed_token_ids.append(token_id)

        return allowed_token_ids
