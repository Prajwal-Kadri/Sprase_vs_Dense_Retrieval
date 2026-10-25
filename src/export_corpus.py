import json
from pathlib import Path

from datasets import load_dataset

corpus = load_dataset("bbasavar/wiki-corpus-v1", split="train")

output_path = Path("data/wiki-corpus-v1.jsonl")

with output_path.open("w", encoding="utf-8") as output:
    for article in corpus:
        row = {
            "title": article["title"],
            "text": article["text"],
        }
        output.write(json.dumps(row) + "\n")