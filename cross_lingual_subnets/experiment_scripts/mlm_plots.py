# get the subnetworks and evaluate on mlm task for each other language

# get each of the pruned subnetworks

import numpy as np
import copy
from torch.utils.data import DataLoader, SequentialSampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

languages = ['ar', 'de', 'en', 'es', 'hi', 'ru', 'ur', 'zh']

results = {} # key: subnetwork, value: list of eval results on other languages

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained('../../models/final_checkpoint')

dataset = get_dataset_head_masks(
    dataset_name="mbelitsky/wikipedia_subset",
    tokenizer=tokenizer,
    n_examples_per_lang=100000,
    seed=1234,
    test_size=3000,
    cache_dir=None,
    languages=languages,
    load_dataset_dict_path=None,
    save_dataset_dict_path='wiki_datasets/'
)


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

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):

        input_ids, input_mask, label_ids = batch["input_ids"], batch["attention_mask"], batch["labels"]
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        label_ids = label_ids.to(args.device)
        
        outputs = model(
            input_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask
        )

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


for l_model in languages:

    cur_mask_path = f'../../old_prune_out/{l_model}_seed_42_head_mask.npy'
    cur_mask = np.load(cur_mask_path)

    cur_model = prune_from_saved_mask(model, cur_mask)
    
    for l_test in languages:

        cur_dataset = dataset['test'][l_test]

        eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
        eval_dataloader = DataLoader(cur_dataset, sampler=eval_sampler, batch_size=4, collate_fn=data_collar)

        cur_perplexity = get_perplexity(cur_model, eval_dataloader)
