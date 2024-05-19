"""
pruning method for sub-networks (WIP)

only pruning attention heads
could be extended to MLP layers
"""

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

from einops import rearrange

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

# prune utils functions -> could be put in prune_utils.py

def entropy(p):
    """Compute the entropy of a probability distribution"""
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

def print_2d_tensor(tensor):
    """Print a 2D tensor"""
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

# TODO: rewrite functions into OOP style
# main functions of pruning

def init_tensors(model, device, head_mask=None, compute_importance=True):
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)

    if head_mask is None and compute_importance:
        head_mask = torch.ones(n_layers, n_heads).to(device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)
    
    return head_importance, attn_entropy, head_mask

def compute_heads_importance(
    args,
    model,
    eval_dataloader,
    compute_entropy=True,
    compute_importance=True,
    head_mask=None
):
    """ 
    This method computes:
        - head attention entropy
        - head importance scores
    """

    # preparing tensors
    head_importance, attn_entropy, head_mask = init_tensors(model, args.device, head_mask, compute_importance)

    preds = None
    labels = None
    tot_tokens = 0.0

    neg_log_likelihood = 0.0

    save_prefix = args.language + '_seed_' + str(args.seed) + "_"

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        
        '''
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        '''

        input_ids, input_mask, label_ids = batch["input_ids"], batch["attention_mask"], batch["labels"]
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        label_ids = label_ids.to(args.device)
        
        '''
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

        print(tokenizer.decode(input_ids[0]))
        print(input_mask[0])
        print(tokenizer.decode(label_ids[0]))
        '''

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(
            input_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask
        )

        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last

        neg_log_likelihood += input_mask.float().detach().sum().data*loss.float().detach().item()

        loss.backward()  # Backpropagate to populate the gradients in the head mask
        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * input_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()
        # Also store our logits/labels if we want to compute metrics afterwards

        '''
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)
        '''

        tot_tokens += input_mask.float().detach().sum().data

    # TODO: separate the following code into smaller functions
    
    if compute_entropy:
        # Normalize
        attn_entropy /= tot_tokens
        np.save(os.path.join(args.output_dir, save_prefix + "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
        logger.info("Attention entropies")
        print_2d_tensor(attn_entropy)
    if compute_importance:
        # Normalize
        head_importance /= tot_tokens
        # Layerwise importance normalization
        if not args.dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not args.dont_normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

        # Print/save matrices
        np.save(os.path.join(args.output_dir, save_prefix + "head_importance.npy"), head_importance.detach().cpu().numpy())

        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
        logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=args.device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print_2d_tensor(head_ranks)

    perplexity = torch.exp(neg_log_likelihood/tot_tokens)
    #print(f"Final perplexity: {perplexity}")
    return attn_entropy, head_importance, perplexity

def compute_metrics(task_name, preds, labels):
    # TODO: implement this function
    # compute metrics for the task
    # perplexity, accuracy, f1, etc.
    if task_name == "mlm":
        return {"perplexity": Perplexity().compute(predictions=preds, references=labels)}

def mask_heads(args, model, eval_dataloader):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    save_prefix = args.language + '_seed_' + str(args.seed) + "_"

    _, head_importance, perplexity = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    original_score = perplexity
    logger.info("Pruning: original score (perplexity): %f, threshold: %f", original_score, original_score * args.masking_threshold)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    current_score = original_score
    i = 0
    while current_score <= original_score * args.masking_threshold:            
        head_mask = new_head_mask.clone()  # save current head mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.config.num_attention_heads
            head_idx = head.item() % model.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())
                
        if not selected_heads_to_mask:
            break

        logger.info("Heads to mask: %s", str(selected_heads_to_mask))
        
        #new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, cur_perplexity = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        current_score = cur_perplexity

        logger.info(
            "Masking: current score: %f, remaining heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )
        

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args.output_dir, save_prefix + "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask

def prune_heads(args, model, eval_dataloader, head_mask):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    #before_time = datetime.now()
    _, _, perplexity = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    score_masking = perplexity
    #original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    logger.info(f"{heads_to_prune}")
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    #before_time = datetime.now()
    _, _, perplexity = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=None
    )
    score_pruning = perplexity
    #new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    #logger.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        # NOTE: now we have one model only, so this can be redundant
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        # if you want to use a model from huggingface use this arg otherwise leave blank
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        # if you want to use a local checkpoint use this arg
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # TODO: should add language into input and output file names
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="The language to use.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--save_mask_all_iterations", action="store_true", help="Saves the masks and importance scores in all iterations"
    )
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )

    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument(
        "--use_train_data", action="store_true", help="Use training set for computing masks"
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default=None, type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()

    # NOTE: distant debugging stuff, might be removed if it works without it
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    set_seed(args.seed, args.n_gpu)

    # NOTE: this could be a bit redundant, we could remove it
    if args.metric_name is None:
        args.metric_name = {
            "mnli": "acc",
            "mlm": "perplexity"
        }[args.task_name]
    

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    assert args.model_name is None or args.model_path is None, "Only one of model_name and model_path should be used"
    
    if args.model_name:
        # Load the tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Load the model
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    if args.model_path:

        # Load the tokenizer (assuming it's going to be xlm-roberta)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

        # Load the model
        model = AutoModelForMaskedLM.from_pretrained(args.model_path)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print/save training arguments
    torch.save(args, os.path.join(args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", args)

    # Prepare dataset for the GLUE task
    # TODO: investigate if we need this load_and_cache_examples function
    # or we could just use the DataLoader directly from the dev set data
    # it probably depends on the task
    #eval_data = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    """
    if args.use_train_data:
        train_data = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        eval_data = random_split(train_data, [true_eval_len, len(train_data) - true_eval_len])[0]
    if args.data_subset > 0:
        eval_data = Subset(eval_data, list(range(min(args.data_subset, len(eval_data)))))
    """

    dataset = DatasetDict.load_from_disk(args.data_dir)

    eval_data = dataset['test']
    eval_data = eval_data.select(range(300))
    
    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    data_collar = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=data_collar)

    

    # Try head masking (set heads to zero until the score goes under a threshold)
    # and head pruning (remove masked heads and see the effect on the network)
    print(args.try_masking)
    if args.try_masking: # remove condition on masking_threshold to be in [0, 1]
        print("Creating head mask")
        head_mask = mask_heads(args, model, eval_dataloader)
        prune_heads(args, model, eval_dataloader, head_mask)
    else:
        #Compute head entropy and importance score
        compute_heads_importance(args, model, eval_dataloader)

if __name__ == "__main__":
    main()