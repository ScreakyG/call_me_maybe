from pydantic import ValidationError
import argparse
import sys
import llm_sdk
import json
from src.json_parsing import parse_input_files

from src.models import PromptInput, FunctionDefinition

# Remove later
import time

DEFAULT_FUNCTIONS_FILE = "data/input/functions_definition.json"
DEFAULT_PROMPTS_FILE = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_FILE = "data/output/function_calling_results.json"

model = llm_sdk.Small_LLM_Model()


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


def generate_next_token(input_ids: list[int], allowed_tokens_ids: list[int] | None = None) -> int:

    logits: list[float] = model.get_logits_from_input_ids(input_ids)
    # print(logits)

    # Should only select tokens from the allowlist
    if allowed_tokens_ids:
        next_token_id = max(
            allowed_tokens_ids,
            key=logits.__getitem__
        )

    else:
        next_token_id = max(
            range(len(logits)),
            key=logits.__getitem__
        )



    print("allowed_token_ids =", allowed_tokens_ids)
    print("next_token_id =", next_token_id)
    print("next_token_decoded =", model.decode([next_token_id]))

    # return allowed_tokens_ids[0]

    return next_token_id



def get_allowed_tokens_ids(function_defs: list[FunctionDefinition]) -> list[int]:

    # boilerplate = '{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers"}'

    # allowed_tokens = boilerplate[position]
    # print("allowed_tokens =", allowed_tokens)

    # allowed_tokens_encoded = model.encode(allowed_tokens)
    # allowed_tokens_ids: list[int] = allowed_tokens_encoded[0].tolist()
    # print("allowed_tokens_ids =", allowed_tokens_ids)


    # Just a test to try if the model can answer only with function names (atm in only allow 'fn')
    allowed_tokens_ids: set[int] = set()
    for function in function_defs:
        function_ids = model.encode(function.name)[0].tolist()

        if function_ids:
            allowed_tokens_ids.add(function_ids[0])

    return list(allowed_tokens_ids)


def build_prompt(functions_def: list[FunctionDefinition], user_prompt: str) -> str:

    prompt_base = (
        "Choose the function that best matches the user request.\n"
        "You must select one function from this provided list: \n"
    )

    prompt_base += "\n".join(function.model_dump_json(indent=2) for function in functions_def)

    prompt_base += f"\n User request: {user_prompt}"

    print(prompt_base)

    return (prompt_base)

def llm_testing(functions_def: list[FunctionDefinition], parsed_prompts: list[PromptInput]) -> None:

    # input_tokens = "What is the captal of France ?"
    input_tokens = build_prompt(functions_def, parsed_prompts[2].prompt)

    encoded = model.encode(input_tokens)
    input_ids: list[int] = encoded[0].tolist()
    # print("liste des tokens_ids:", input_ids)


    output_tokens = ""

    while True:

        print("\n==================================\n")
        # print("Current completion = ", model.decode(input_ids))

        # allowed_tokens_ids = get_allowed_tokens_ids(functions_def)
        allowed_tokens_ids = None

        next_token_id = generate_next_token(input_ids, allowed_tokens_ids)
        input_ids.append(next_token_id)


        output_tokens += model.decode([next_token_id])
        print(output_tokens)


        time.sleep(1) # Remove this later , just to slow down generation since its going too fast now

def main() -> None:
    try:
        file_config = parse_args()
        functions_defs, parsed_prompts = parse_input_files(file_config)

        # print("functions_defs:", functions_defs)
        # print()
        # print("prompts:", parsed_prompts)

    except OSError as error:
        print(f"File not found: {error.filename}", file=sys.stderr)
        print(error, file=sys.stderr)

    except json.decoder.JSONDecodeError as error:
        print(f"Invalid JSON syntax: {error} {error.doc}", file=sys.stderr)

    except ValidationError as error:
        print("Invalid JSON data structure:", file=sys.stderr)
        print(error, file=sys.stderr)


    llm_testing(functions_defs, parsed_prompts)

    # vocab_file = model.get_path_to_vocab_file()
    # with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as file:
        # vocab: dict[str, int] = json.load(file)

    # print(vocab)
    # print(vocab['Hello'])



if __name__ == "__main__":
    main()
