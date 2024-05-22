import json
import logging

import pandas as pd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

# Load the data
with open("data/bible_parallel_corpus.json", "r", encoding="ISO-8859-1") as f:
    texts = json.load(f)

# Compute the mean per sentence intersection of tokens between english and other languages
target_lang = "en"
per_lang_intersection_percentages = {}

for sentences in texts:
    target_tokens = set(tokenizer(sentences[target_lang])["input_ids"])
    for lang in sentences:
        if lang == target_lang:
            continue
        lang_tokens = set(tokenizer(sentences[lang])["input_ids"])
        intersection = target_tokens.intersection(lang_tokens)
        per_lang_intersection_percentages[lang] = per_lang_intersection_percentages.get(
            lang, 0
        ) + len(intersection) / len(lang_tokens)

# Aggregate the results
intersection_percentages = {
    lang: pct / len(texts) for lang, pct in per_lang_intersection_percentages.items()
}

print(
    pd.DataFrame(intersection_percentages, index=["Mean"]).T.sort_values(
        "Mean", ascending=False
    )
)
