import argparse

from cross_lingual_subnets.prune import prune_from_saved_mask

from transformers import AutoModelForMaskedLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get pruned model given head mask")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to checkpoint for pruning"
    )
    parser.add_argument(
        "--mask_path", type=str, required=True, help="Path to head mask to use"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)

    original_num_params = sum(p.numel() for p in model.parameters())
    pruned_model = prune_from_saved_mask(args, model)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    print(f"Original parameters: {original_num_params}")
    print(f"Pruned parameters: {pruned_num_params}")
    print(
        "Eliminated parameters:"
        f" {(original_num_params - pruned_num_params) / original_num_params * 100:.2f}%"
    )
