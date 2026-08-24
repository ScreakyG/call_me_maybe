from src.models import PromptInput, FunctionDefinition
import json


# Try to open input files and validate the json_content
def parse_input_files(file_config: dict[str, str]) -> tuple[list[PromptInput], list[FunctionDefinition]]:

    parsed_prompts: list[PromptInput] = []
    functions_defs: list[FunctionDefinition] = []

    for key, value in file_config.items():

        if key in ['functions_definition', 'input']:
            with open(value, "r") as file:
                # parse_json_file(key, value, file)

                data = json.load(file)
                for items in data:
                    match (key):
                        case 'functions_definition':
                            parsed_data = FunctionDefinition.model_validate(items)
                            functions_defs.append(parsed_data)

                        case 'input':
                            parsed_data = PromptInput.model_validate(items)
                            parsed_prompts.append(parsed_data)

    return functions_defs, parsed_prompts
