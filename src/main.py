import sys

DEFAULT_FUNCTIONS_FILE = "data/input/functions_definition.json"
DEFAULT_PROMPTS_FILE = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_FILE = "data/output/function_calls.json"

file_config = {
    "functions_definition": DEFAULT_FUNCTIONS_FILE,
    "input": DEFAULT_PROMPTS_FILE,
    "output": DEFAULT_OUTPUT_FILE
}


def load_input_files() -> None:
    print(f"Functions definition file: {file_config['functions_definition']}")
    print(f"Prompt inputs file: {file_config['input']}")
    print(f"Output file: {file_config['output']}")


def parse_args() -> None:
    # Try to override the default config with provided args
    if len(sys.argv) > 1:
        try:
            for i in range(1, len(sys.argv), 2):
                argument = sys.argv[i]
                argument_name = argument[2:]

                if (
                    argument.startswith("--", 0, 2)
                    and argument_name in file_config.keys()
                ):
                    if (
                        len(sys.argv) > i + 1
                        and not sys.argv[i + 1].startswith("--", 0, 2)
                    ):
                        file_config[argument_name] = sys.argv[i + 1]
                    else:
                        raise Exception(f"Missing value for {argument}")
                else:
                    raise Exception(f"Argument '{argument}' is invalid")

        except Exception as error:
            print(error, file=sys.stderr)
            sys.exit(1)


def main() -> None:
    parse_args()
    load_input_files()


if __name__ == "__main__":
    main()
