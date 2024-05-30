import os
import argparse

from cross_lingual_subnets.prune import prune_from_saved_mask
from scripts.create_subsets import WIKIPEDIA_DUMPS

from transformers import AutoModelForMaskedLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get pruned model given head mask")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to checkpoint for pruning"
    )
    parser.add_argument(
        "--mask_dir", type=str, required=True, help="Path to head mask to use"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save pruned model"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="*",
        required=True,
        choices=WIKIPEDIA_DUMPS.keys(),
        help=(
            "The languages for creating subnetworks"
        ),
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)

    for lang in args.languages:
        mask_path = os.path.join(args.mask_dir, f"{lang}_seed_42_head_mask.npy")
        # Prune the model
        pruned_model = prune_from_saved_mask(model, mask_path=mask_path)
        # Save the pruned model
        save_path = os.path.join(args.output_dir, f"pruned_{lang}_mlm_finetuned")
        pruned_model.save_pretrained(save_path)
