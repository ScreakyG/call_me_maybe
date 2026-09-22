*This project has been created as part of the 42 curriculum by fgonzale.*

# Call Me Maybe

## Description

Call Me Maybe translates natural language requests into structured function calls using a local language model. Given a prompt and a list of function definitions, it generates a JSON object containing the original prompt, the selected function name, and its parameters. It does not execute the function.

The goal is to learn how constrained decoding works by implementing it from scratch: at each generation step, only tokens compatible with the expected structure are allowed. The project uses `Qwen/Qwen3-0.6B` through the provided `llm_sdk`, without a structured-output framework.

This project was made for learning purposes and is not ready for production use.

## Instructions

Requirements: Python 3.12 or later, `uv` and optionally `make`. Run the commands from the repository root. Keep the provided `llm_sdk/` directory alongside `src/`.

Install dependencies and run:

```bash
uv sync
uv run python -m src
```

Or use the Makefile:

```bash
make install
make run
```

The first run needs internet access to download the model and tokenizer. The SDK automatically selects MPS or CUDA when available, otherwise it uses the CPU.

Default files:

- Function definitions: `data/input/functions_definition.json`
- Prompts: `data/input/function_calling_tests.json`
- Results: `data/output/function_calling_results.json`

You can change these files by providing arguments :
`uv run python -m src [–functions_definition <function_definition_file>]
[–input <input_file>] [–output <output_file>]`

Both input files contain JSON arrays. Each prompt has a `prompt` field. Each function has a `name`, `description`, `parameters`, and `returns` definition, as shown in the provided input files. Supported parameter types are `string`, `number`, and `boolean`; the current input schema requires at least one parameter per function.

The output directory is created if needed. An existing output file is overwritten. The program currently prints detailed token generation information to the terminal.

Other available commands:

- `make debug`: run with Python's `pdb` debugger.
- `make lint`: run flake8 and mypy.
- `make lint-strict`: run flake8 and mypy in strict mode.
- `make clean`: remove Python and mypy caches.

## Algorithm explanation

Generation follows these steps for each prompt:

1. Read the input files and validate prompts and function definitions with Pydantic.
2. Build a model prompt containing the available function definitions and the user request, then encode it into token IDs.
3. Use a state machine to track the expected JSON structure and find compatible tokens in the model vocabulary.
4. Request the next-token logits from the SDK. Set disallowed tokens' logits to negative infinity, then apply softmax to obtain probabilities.
5. Sample a token from the allowed distribution, append it to the model input, and advance the state machine.
6. Repeat until the object is complete, then parse the generated JSON and save the collected results in a JSON array.

There are three main parts to the state machine:

- `Automate` handles the outer object and reproduces the original prompt with JSON escaping.
- `GenerationState` restricts function names to prefixes of names present in the input definitions. A name can end only when it matches an available function.
- `ParametersAutomate` builds the parameter structure from the selected function. Parameter names and punctuation are constrained, while values follow rules for strings, numbers, and booleans.

Parameter tokens are checked character by character on a copy of the state before being accepted. This allows a single token to span several parts of the parameter structure. String handling checks escapes, including Unicode escapes; number handling accepts valid prefixes until the number is complete; booleans are limited to `true` and `false`.

These constraints control the output format. Choosing the appropriate function and extracting the right values still depend on the model.

## Design decisions

- Separate input parsing, generation, and state handling into `json_parsing.py`, `main.py`, and `models.py`.
- Build constraints from the supplied function definitions so function names and parameter names are not hardcoded.
- Use the public SDK methods for model inference and tokenization, Pydantic for input validation, and NumPy for probability calculations and sampling.
- Sample from the constrained distribution rather than always selecting the highest-scoring token. Results can therefore vary between runs.
- First i was using a `greedy decoding` solution to get the token but then i was skipping what the subject wanted , however when the goal is to have structured output it might be better to have the
highest ranked token to be selected. With the latest solution i used i could have implemented it back.

## Performance analysis

The current implementation scans and decodes vocabulary tokens at every generation step. It also calls the model for each new token and prints detailed diagnostics. These operations contribute to runtime, which depends on the hardware and prompt/output length.

No measured benchmark is documented here yet. The subject's targets are at least 90% correct function selection and argument extraction, 100% valid and schema-compliant JSON, and processing the test prompts in under five minutes. These are targets, not measured results.

The tests ran on my personal machine , using `RTX 5090 / AMD 9800x3d / 32GB DDR5`
Tested with the prompts from the subject and my personals prompts

**TODO :** number of prompts, total runtime, proportion of fully correct calls, and proportion of valid JSON and schema-compliant outputs. Mention remaining failures and whether timing includes model loading. Repeat runs to account for sampling variability.

## Challenges faced

- Parse user's JSON using `pydantic`
- How to handle correctly parameters terminaison, this was the hardest part because first i was allowed the model to only use `"` to terminate `STRING` parameters which was a problem because the model would sometimes use tokens like `", `.
- When constrainted too much the model could be stuck in infinite loops
- Same thing with `NUMBER` parameters terminaison using `,`


## Testing strategy

The provided prompts cover addition, greetings, string reversal, square roots, and regex replacement. To validate a run:

1. Check that the output parses as JSON and contains one result per input prompt.
2. Check the `prompt`, `name`, and `parameters` fields, the selected function, and the exact parameter names and types against its definition.
3. Compare argument values with the original request: valid JSON alone does not prove that a call is correct.
4. Try other function definitions and edge cases, such as escaped characters, empty string arguments, negative or decimal numbers, booleans, and ambiguous requests.
5. Check how missing files, malformed JSON, and invalid input structures are reported.


## Example usage

Run with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Pass options through the Makefile:

```bash
make run ARGS="--input data/input/function_calling_tests.json --output data/output/example_results.json"
```

For `What is the sum of 2 and 3?`, the expected output format is:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
  }
]
```

This is an illustrative expected result, not a recorded model run. The program processes every prompt in the input file.

## Resources

- [Project subject](en.subject.pdf): requirements and introduction to constrained decoding.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): information about the model used by the SDK.
- [Pydantic documentation](https://docs.pydantic.dev/latest/): input data validation.
- [uv documentation](https://docs.astral.sh/uv/): dependency and environment management.
- [DeepLearning.AI](https://www.deeplearning.ai/courses/getting-structured-llm-output).

### AI usage

- AI assisted with drafting this README from the project subject and the current source code.
- Mostly to ask questions about how things works (logits, constrained-decoding, temperature, top-k, top-p)
- I would always implement things myself first and then use a agent to review my code and see if it can find lacking features, potential bugs
