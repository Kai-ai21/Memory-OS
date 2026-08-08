"""Token counting.

A heuristic, isolated behind one function so M1.5 can swap in the embedding
model's real tokenizer if measurement ever shows the approximation matters.

Deliberately not tiktoken: that is OpenAI's BPE, and the model here is a
sentence-transformer with a different vocabulary. A wrong tokenizer's exact
count is not better than an approximate one, and it would be a dependency
pretending to precision it does not have.
"""

import re

# Words, numbers, and standalone punctuation. Sub-word splitting is what a real
# tokenizer adds; for sizing a chunk, counting words and punctuation lands
# within about 25% of a sentence-transformer's count, which is well inside the
# margin a 640-token target already tolerates.
_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))
