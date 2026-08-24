import argparse
import sys
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, Field
from enum import Enum
import llm_sdk


DEFAULT_FUNCTIONS_FILE = "data/input/functions_definition.json"
DEFAULT_PROMPTS_FILE = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_FILE = "data/output/function_calling_results.json"


class PromptsInputs(BaseModel):
    prompt: str = Field(min_length=1)

class ParametersType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOL = "bool"

class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ParametersType

# class ReturnsDefinition(BaseModel):
#     model_config = ConfigDict

class FunctionsDefinitions(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterDefinition] = Field(min_length=1)
    returns: ParameterDefinition


parsed_prompts: list[PromptsInputs] = []
functions_defs: list[FunctionsDefinitions] = []

# Verify that input files are json and that they contains exepected data
def parse_json_file(file_type: str, filename: str, file: Any) -> None:

    data = json.load(file)

    try:

        for items in data:
            match (file_type):
                case 'functions_definition':
                    parsed_data = FunctionsDefinitions.model_validate(items)
                    # print(parsed_data)
                    functions_defs.append(parsed_data)

                case 'input':
                    parsed_data = PromptsInputs.model_validate(items)
                    parsed_prompts.append(parsed_data)
                    # print(parsed_data)

    except ValidationError as error:
        print(f"JSON Schema is not valid for {file_type} file in : '{filename}'")
        print(error)

# Open the input files
def load_input_files(file_config: dict[str, str]) -> None:
    # print(f"Functions definition file: {file_config['functions_definition']}")
    # print(f"Prompt inputs file: {file_config['input']}")
    # print(f"Output file: {file_config['output']}")

    for key, value in file_config.items():
        try:

            # print(key, value)
            if key in ['functions_definition', 'input']:
                with open(value, "r") as file:
                    parse_json_file(key, value, file)

        except OSError as error:
            print(error, file=sys.stderr)

        except json.decoder.JSONDecodeError:
            print(f"Error: Invalid JSON format in the file: {value}")

# Verify if arguments where provided and parse them
def parse_args(argv: list[str] | None = None) -> dict[str, str]:
    parser = argparse.ArgumentParser(
        prog="Call Me Maybe",
        description="Translate prompts into functions calls",
        usage=(
            "[--functions_definition <function_definition_file>]"
            "[--input <input_file>]"
            "[--output <output_file>]"
        )
    )

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_FILE,
        help="<path/to/functions_definition>"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_PROMPTS_FILE,
        help="<path/to/input_file>"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="<path/to/output_file>"
    )

    return vars(parser.parse_args(argv))


def main() -> None:

    file_config = parse_args()
    load_input_files(file_config)

    print("functions_defs:", functions_defs)
    print()
    print("prompts:", parsed_prompts)

    # model = llm_sdk.Small_LLM_Model()
    # print(model)
    # result = model.encode("Hello World one more token,2")
    # print(result)


if __name__ == "__main__":
    main()
