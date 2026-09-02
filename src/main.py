from pydantic import ValidationError
import argparse
import sys
import llm_sdk
import json
from src.json_parsing import parse_input_files

import numpy as np
from src.models import PromptInput, FunctionDefinition, Automate

# Remove later
import time

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

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

    if not allowed_tokens_ids:
        raise Exception("There is no allowed tokens")

    logits: list[float] = model.get_logits_from_input_ids(input_ids)
    # print(logits)


    # Create a numpy array from logits list with float type values
    logits_array = np.asarray(logits, dtype=float)
    # print("logits_array =", logits_array)

    # Create a array with the same shape as the src and fill values with -inf
    masked_logits = np.full_like(logits_array, -np.inf)
    # print("masked_logits=", masked_logits)

    # Put back the og logit value for allowed_tokens_ids, NumPy allows to use list of indices perform this easily
    masked_logits[allowed_tokens_ids] = logits_array[allowed_tokens_ids]
    # print("allowed_tokens_logits=", masked_logits)

    # We use exp to transform all logits to be superior or equal to 0 while preserving the gap between them
    weights = np.exp(masked_logits)
    # print("weights=", weights)

    # We normalize weights to produce probabilities numbers
    total = np.sum(weights)
    probabilities = weights / total
    # print("probabilities=", probabilities)

    # Sampling: get a token_id from the calculated probabilities
    next_token_id = np.random.choice(
        len(probabilities),
        p=probabilities
    )


    for token_id in allowed_tokens_ids:
        print(
            f"id: {token_id} | "
            f"decoded: {model.decode([token_id])} | "
            f"prob: {probabilities[token_id]:.6%}"
        )

    print(f"{GREEN}next_token_id =", next_token_id)
    print(f"next_token_decoded = {model.decode([next_token_id])}{RESET}")

    return next_token_id


def build_prompt(functions_def: list[FunctionDefinition], user_prompt: str) -> str:

    prompt_base = (
        "Choose the function that best matches the user request.\n"
        "You must select one function from this provided list:\n"\
    )

    prompt_base += "\n".join(function.model_dump_json(indent=2) for function in functions_def)

    prompt_base += f"\n User request: {user_prompt}"

    # print(prompt_base)

    return (prompt_base)


def get_vocab_token_ids() -> list[int]:
    vocab_file = model.get_path_to_vocab_file()
    tokens_ids: list[int] = []

    with open(vocab_file, "r") as file:
        json_file = json.load(file)

        tokens_ids = json_file.values()

    return list(tokens_ids)



def llm_testing(functions_def: list[FunctionDefinition], parsed_prompts: list[PromptInput]) -> None:

    input_tokens = build_prompt(functions_def, parsed_prompts[3].prompt + "\n")
    encoded = model.encode(input_tokens)
    input_ids: list[int] = encoded[0].tolist()


    output_tokens = ""

    vocab_token_ids = get_vocab_token_ids()
    automate = Automate(model, parsed_prompts[3].prompt, vocab_token_ids, functions_def)

    while not automate.stop_sequence():

        print("\n==================================\n")
        # print("Current completion = ", model.decode(input_ids))

        # Get allowed token_ids for current sequence
        allowed_tokens_ids = automate.get_current_sequence_allowed_tokens()

        # Produce logits with only those allowed ids
        next_token_id = generate_next_token(input_ids, allowed_tokens_ids)

        # Decode the generated next_token_id to see the text representation
        decoded_id = model.decode([next_token_id])

        # Check if generated token can be to current sequence
        automate.append_to_sequence(decoded_id)

        # Append the generated token to the completion string
        input_ids.append(next_token_id)

        # String to show what has been completed by the LLM
        output_tokens += decoded_id
        print(output_tokens)

        # Remove this later , just to slow down generation since its going too fast now
        time.sleep(0.2)

        # Try proceed to next sequence if the current one is completed
        automate.increase_sequence()

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
