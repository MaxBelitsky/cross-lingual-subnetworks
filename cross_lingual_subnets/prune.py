import torch
import numpy as np
import copy


def prune_from_saved_mask(model, head_mask=None, mask_path: str | None = None):
    """This loads the head mask and returns the pruned model"""
    if head_mask is None:
        head_mask = torch.tensor(np.load(mask_path))

    # To not modify the original one
    model = copy.deepcopy(model)

    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask

    assert (
        sum(len(h) for h in heads_to_prune.values())
        == (1 - head_mask.long()).sum().item()
    )
    model.prune_heads(heads_to_prune)

    return model
