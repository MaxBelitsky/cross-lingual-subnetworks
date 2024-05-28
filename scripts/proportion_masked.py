import numpy as np
import os

BASE_PRUNE_PATH = "../outputs/old_prune_output/"
LANGUAGES = ["ar", "de", "en", "es", "hi", "ru", "ur", "zh"]


def compute_proportion_masked():
    # first dim is layer
    mask = [None] * 8
    for i, lang in enumerate(LANGUAGES):
        mask[i] = np.load(
            os.path.join(BASE_PRUNE_PATH, f"{lang}_seed_42_head_mask.npy")
        )

    masked_by_layer = np.zeros(12)

    for m in mask:
        print(m)
        cur_mask_by_layer = 12 - np.sum(m, axis=0)
        print(cur_mask_by_layer)
        masked_by_layer += cur_mask_by_layer

    masked_by_layer = masked_by_layer / 8 / 12

    print(masked_by_layer)


if __name__ == "__main__":
    compute_proportion_masked()
