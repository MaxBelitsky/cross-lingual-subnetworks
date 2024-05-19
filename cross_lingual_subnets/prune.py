import argparse
import logging
import os
import random

from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torcheval.metrics.text import Perplexity
import numpy as np

from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from transformers import glue_processors as processors

from datasets import load_dataset, DatasetDict, concatenate_datasets

from torch import Tensor

def prune_from_saved_mask(args, model):
    """ This loads the head mask and returns the pruned model
    """
    head_mask = torch.tensor(np.load(args.mask_path))
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    model.prune_heads(heads_to_prune)
    
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get pruned model given head mask")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to checkpoint for pruning"
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to head mask to use"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)

    original_num_params = sum(p.numel() for p in model.parameters())

    pruned_model = prune_from_saved_mask(args, model)

    pruned_num_params = sum(p.numel() for p in model.parameters())

    print(f"Original parameters: {original_num_params}")
    print(f"Pruned parameters: {pruned_num_params}")
    print(f"Eliminated parameters: {original_num_params - pruned_num_params}")

    