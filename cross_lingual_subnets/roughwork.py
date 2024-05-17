
from cross_lingual_subnets.data import get_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the tokenizerx
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

# Load the model
model = AutoModelForMaskedLM.from_pretrained('FacebookAI/xlm-roberta-base')


# Load the dataset

dataset = get_dataset(
    dataset_name="mbelitsky/wikipedia_subset",
    tokenizer=tokenizer,
    n_examples_per_lang=100000,
    seed=1234,
    test_size=3000,
    cache_dir=None,
    languages=['lt', 'ru', 'en', 'fr', 'de', 'zh', 'sw', 'ar', 'hi', 'fa', 'ur', 'es'],
)

