from pydantic import ValidationError
import argparse
import sys
# import llm_sdk
import json
from src.json_parsing import parse_input_files

DEFAULT_FUNCTIONS_FILE = "data/input/functions_definition.json"
DEFAULT_PROMPTS_FILE = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_FILE = "data/output/function_calling_results.json"


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
    try:
        file_config = parse_args()
        functions_defs, parsed_prompts = parse_input_files(file_config)

        print("functions_defs:", functions_defs)
        print()
        print("prompts:", parsed_prompts)

    except OSError as error:
        print(f"File not found: {error.filename}", file=sys.stderr)
        print(error, file=sys.stderr)

    except json.decoder.JSONDecodeError as error:
        print(f"Invalid JSON syntax: {error} {error.doc}", file=sys.stderr)

    except ValidationError as error:
        print("Invalid JSON data structure:", file=sys.stderr)
        print(error, file=sys.stderr)

    # model = llm_sdk.Small_LLM_Model()
    # print(model)
    # result = model.encode("Hello World one more token,2")
    # print(result)


if __name__ == "__main__":
    main()
