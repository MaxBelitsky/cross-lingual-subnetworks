
from cross_lingual_subnets.data import get_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse

# Load the tokenizerx
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

parser = argparse.ArgumentParser(description="Training sentence embedding models")

parser.add_argument(
    "--languages",
    type=str,
    nargs="*",
    help="The languages to include in the dataset. If not provided, all languages are included."
)

args = parser.parse_args()

# Load the dataset

dataset = get_dataset(
    dataset_name="mbelitsky/wikipedia_subset",
    tokenizer=tokenizer,
    n_examples_per_lang=100000,
    seed=1234,
    test_size=3000,
    cache_dir=None,
    languages=args.languages,
    load_dataset_dict_path=None,
    save_dataset_dict_path='wiki_datasets/'
)

