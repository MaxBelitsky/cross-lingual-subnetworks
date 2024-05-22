# get the subnetworks and evaluate on mlm task for each other language

# get each of the pruned subnetworks

import argparse
import logging
import os
import random

from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from transformers import glue_processors as processors

from datasets import load_dataset, DatasetDict, concatenate_datasets

from torch import Tensor

import os
import logging
from itertools import chain

from datasets import load_dataset, DatasetDict, concatenate_datasets

import copy

logger = logging.getLogger(__name__)

languages = ['ar', 'de', 'en', 'es', 'hi', 'ru', 'ur', 'zh']

results = {} # key: subnetwork, value: list of eval results on other languages

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained('/content/drive/MyDrive/ATCS_checkpoint/final_checkpoint').to('cuda')


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

    if True:
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


def prune_from_saved_mask(model, head_mask):
    """ pasting this code and editing it slightly
    """
    model = copy.deepcopy(model)

    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    model.prune_heads(heads_to_prune)
    
    return model


def get_perplexity(model, eval_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neg_log_likelihood = 0.0
    tot_tokens = 0.0

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        
        input_ids, input_mask, label_ids = batch["input_ids"], batch["attention_mask"], batch["labels"]
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)
        
        outputs = model(
            input_ids, attention_mask=input_mask, labels=label_ids)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last

        neg_log_likelihood += input_mask.float().detach().sum().data*loss.float().detach().item()

        tot_tokens += input_mask.float().detach().sum().data

    perplexity = torch.exp(neg_log_likelihood/tot_tokens)

    return perplexity



data_collar = DataCollatorForLanguageModeling(tokenizer=tokenizer)

datasets_by_l = {}

for l in languages:

    cur_dataset = get_dataset_head_masks(
      dataset_name="mbelitsky/wikipedia_subset",
      tokenizer=tokenizer,
      n_examples_per_lang=100000,
      seed=1234,
      test_size=3000,
      cache_dir=None,
      languages=l,
      load_dataset_dict_path=f'wiki_datasets/{l}',
      save_dataset_dict_path=None
    )

    datasets_by_l[l] = cur_dataset['test']
  



for l_model in languages:

    cur_mask_path = f'/content/cross-lingual-subnetworks/old_prune_output/{l_model}_seed_42_head_mask.npy'
    cur_mask = torch.tensor(np.load(cur_mask_path)).to('cuda')

    cur_model = prune_from_saved_mask(model, cur_mask)
    
    for l_test in languages:

        cur_dataset = datasets_by_l[l_test]

        eval_sampler = SequentialSampler(cur_dataset)
        eval_dataloader = DataLoader(cur_dataset, sampler=eval_sampler, batch_size=2, collate_fn=data_collar)

        cur_perplexity = get_perplexity(cur_model, eval_dataloader)
