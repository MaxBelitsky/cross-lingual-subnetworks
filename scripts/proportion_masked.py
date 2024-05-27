import numpy as np

# first dim is layer

mask = [None] * 8

mask[0] = np.load("../../old_prune_output/ar_seed_42_head_mask.npy")
mask[1] = np.load("../../old_prune_output/de_seed_42_head_mask.npy")
mask[2] = np.load("../../old_prune_output/en_seed_42_head_mask.npy")
mask[3] = np.load("../../old_prune_output/es_seed_42_head_mask.npy")
mask[4] = np.load("../../old_prune_output/hi_seed_42_head_mask.npy")
mask[5] = np.load("../../old_prune_output/ru_seed_42_head_mask.npy")
mask[6] = np.load("../../old_prune_output/ur_seed_42_head_mask.npy")
mask[7] = np.load("../../old_prune_output/zh_seed_42_head_mask.npy")

masked_by_layer = np.zeros(12)

for m in mask:
    print(m)
    cur_mask_by_layer = 12 - np.sum(m, axis=0)
    print(cur_mask_by_layer)
    masked_by_layer += cur_mask_by_layer

masked_by_layer = masked_by_layer / 8 / 12

print(masked_by_layer)
