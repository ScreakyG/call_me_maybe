from src.models import PromptInput, FunctionDefinition
import json


# Try to open input files and validate the json_content
def parse_input_files(
        file_config: dict[str, str]
) -> tuple[list[FunctionDefinition], list[PromptInput]]:

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
                            function = FunctionDefinition.model_validate(items)
                            functions_defs.append(function)

                        case 'input':
                            prompt = PromptInput.model_validate(items)
                            parsed_prompts.append(prompt)

    return functions_defs, parsed_prompts
