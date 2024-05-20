import os
import logging
from itertools import chain

from datasets import load_dataset, DatasetDict, concatenate_datasets

from cross_lingual_subnets.constants import Datasets

logger = logging.getLogger(__name__)


def get_dataset(
    dataset_name,
    tokenizer,
    n_examples_per_lang=100_000,
    seed=42,
    test_size=3000,
    cache_dir=None,
    languages=None,
):
    """ 
    Load and preprocess the dataset.

    Args:
        dataset_name (str): The name of the dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        n_examples_per_lang (int): The maximum number of examples per language to keep.
        seed (int): The random seed.
        test_size (int): The size of the test set.
        cahce_dir (str): The cache directory.
        languages (list): The list of languages to include.

    Returns:
        datasets.DatasetDict: The preprocessed dataset.
    """
    # Load the dataset
    logger.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir, num_proc=os.cpu_count())

    if dataset_name == Datasets.WIKIPEDIA:
        # Filter languages
        if languages:
            logger.info(f"Filtering languages: {languages}")
            dataset = DatasetDict({lang: dataset[lang] for lang in languages})

        # Tokenize the dataset
        logger.info("Tokenizing the dataset")
        dataset = dataset.map(
            lambda x: tokenizer(x["text"]),
            batched=True,
            remove_columns="text"
        )
        # Chunk the dataset
        logger.info("Chunking the dataset")
        dataset = dataset.map(chunk_texts, batched=True)

        # Downsample and split each language subset
        logger.info(f"Downsampling to {n_examples_per_lang} examples per language")
        dataset = DatasetDict(
            {
                lang: (
                    dataset[lang].select(range(n_examples_per_lang))
                    if len(dataset[lang]) > n_examples_per_lang
                    else dataset[lang]
                )
                for lang in dataset
            }
        )
        logger.info(f"Splitting the dataset with test size {test_size}")
        dataset = DatasetDict(
            {
                lang: dataset[lang].train_test_split(
                    test_size=test_size, shuffle=True, seed=seed
                )
                for lang in dataset
            }
        )

        # Combine all languages
        train_split = concatenate_datasets([dataset[lang]['train'] for lang in dataset], axis=0)
        train_split = train_split.shuffle(seed=seed)
        dataset = DatasetDict({"train": train_split, "test": DatasetDict({lang: dataset[lang]['test'] for lang in dataset})})

    return dataset



def get_dataset_head_masks(
    dataset_name,
    tokenizer,
    n_examples_per_lang=100000,
    seed=42,
    test_size=3000,
    cache_dir=None,
    languages=None,
    load_dataset_dict_path=None,
    save_dataset_dict_path=None
):
    """ 
    Load and preprocess the dataset.

    Args:
        dataset_name (str): The name of the dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        n_examples_per_lang (int): The maximum number of examples per language to keep.
        seed (int): The random seed.
        test_size (int): The size of the test set.
        cahce_dir (str): The cache directory.
        languages (list): The list of languages to include.

    Returns:
        datasets.DatasetDict: The preprocessed dataset.
    """

    if load_dataset_dict_path:
      
      dataset = DatasetDict()
      dataset = dataset.load_from_disk(load_dataset_dict_path)
      return dataset

    # Load the dataset
    logger.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    if dataset_name == Datasets.WIKIPEDIA:
        # Filter languages
        if languages:
            logger.info(f"Filtering languages: {languages}")
            dataset = DatasetDict({lang: dataset[lang] for lang in languages})

        # Tokenize the dataset
        logger.info("Tokenizing the dataset")
        dataset = dataset.map(
            lambda x: tokenizer(x["text"]),
            batched=True,
            remove_columns="text"
        )
        # Chunk the dataset
        logger.info("Chunking the dataset")
        dataset = dataset.map(chunk_texts, batched=True)

        # Downsample and split each language subset
        logger.info(f"Downsampling to {n_examples_per_lang} examples per language")
        dataset = DatasetDict(
            {
                lang: (
                    dataset[lang].select(range(n_examples_per_lang))
                    if len(dataset[lang]) > n_examples_per_lang
                    else dataset[lang]
                )
                for lang in dataset
            }
        )
        logger.info(f"Splitting the dataset with test size {test_size}")
        dataset = DatasetDict(
            {
                lang: dataset[lang].train_test_split(
                    test_size=test_size, shuffle=True, seed=seed
                )
                for lang in dataset
            }
        )

        # Combine all languages
        '''
        train_split = concatenate_datasets([dataset[lang]['train'] for lang in dataset], axis=0)
        train_split = train_split.shuffle(seed=seed)
        dataset = DatasetDict({"train": DatasetDict({lang: dataset[lang]['train'] for lang in dataset}), "test": DatasetDict({lang: dataset[lang]['test'] for lang in dataset})})
        '''
        # save languages separately within datasets dir

        if save_dataset_dict_path:
          
          for lang in languages:
              cur_path = f"{save_dataset_dict_path}/{lang}"
              cur_dataset = DatasetDict({"train": dataset[lang]['train'], "test": dataset[lang]['test']})
              cur_dataset.save_to_disk(cur_path)
        

    return dataset


def chunk_texts(examples, chunk_size=512):
    """
    Chunk the texts into chunks of size chunk_size to prepare for MLM training.
    
    Args:
        examples (dict): The examples to chunk.
        chunk_size (int): The size of the chunks.

    Returns:
        dict: The chunked examples.
    """
    # Concatenate all texts
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
