# Call Me Maybe
- Project focusing on function calling with llms

## Tools
- uv

## Packages
- numpy
- flake8
- mypy


## TODO:
- Parameters function appears to work but when the LLM procudes a STRING parameters it can easily go wrong because all tokens are allowed, we may need to restrict only to some tokens or maybe apply greedy decoding only on this part.
- On NUMBER sequence we could improve how allowed tokens are returned , for now its very basic
- Continue watching the deeplearning about llms outputs
- Prompting the LLM to generate a JSON structure is best pratice but at the moment it seems to make the performance worse and output is worse
- Most of the time the problem happens on prompt 8, the LLM generate repetitive tokens and is kind of stuck in a loop
