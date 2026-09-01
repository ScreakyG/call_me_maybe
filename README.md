# Call Me Maybe
- Project focusing on function calling with llms

## Tools
- uv

## Packages
- numpy
- flake8
- mypy


## TODO:
- At the moment if 2 functions have the same start like "fn_greet" and "fn_greet_shrek" the current implementation will move to the next sequence if the generated name is matching a function name, it means that in this example "fn_greet_shrek" would never be picked , this needs to be fixed
- Previous is partially fixed but we still have a error, the model would always choose the longest function name even if it should not , we need to find a better way
- We also need to change to way to tokens are generated using the solution the subject gives us , by putting -inf to unallowed tokens
