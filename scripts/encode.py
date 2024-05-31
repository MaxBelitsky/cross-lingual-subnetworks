import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, XLMRobertaModel, XLMRobertaConfig
from tqdm.auto import tqdm

from cross_lingual_subnets.constants import Experiments
from cross_lingual_subnets.utils import mean_pooling
from cross_lingual_subnets.data import get_bible_dataloaders_by_language
from scripts.create_subsets import WIKIPEDIA_DUMPS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
experiment_map = {
    "ar": Experiments.AR_SUB_MLM_FINETUNED,
    "de": Experiments.DE_SUB_MLM_FINETUNED,
    "en": Experiments.EN_SUB_MLM_FINETUNED,
    "es": Experiments.ES_SUB_MLM_FINETUNED,
    "hi": Experiments.HI_SUB_MLM_FINETUNED,
    "ru": Experiments.RU_SUB_MLM_FINETUNED,
    "zh": Experiments.ZH_SUB_MLM_FINETUNED,
}


def encode(
    dataloader: DataLoader,
    model,
    language: str,
    output_dir: str,
    experiment_name,
):
    # Save hidden outputs per layer
    hids = {}
    for i, batch in enumerate(tqdm(dataloader)):

        # Note! First layer here is embedding layer
        with torch.no_grad():
            output = model(**batch, output_hidden_states=True)["hidden_states"]

        # Aggregate averages of sentences (hence mean) per layer
        for j in range(len(output)):
            out = mean_pooling(output[j], batch["attention_mask"])
            if i == 0:
                hids[j] = out
            else:
                hids[j] = torch.cat((hids[j], out), 0)

    # Save encodings
    os.makedirs(f"{output_dir}/{experiment_name}/", exist_ok=True)
    save_path = f"{output_dir}/{experiment_name}/{language}.pt"
    print(f"Saving encodings to {save_path}")
    torch.save(hids, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoding sentences using language models"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size of the dataloader",
        default=32,
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        help="The amount of sentences to encode",
        default=1000,
    )
    parser.add_argument(
        "--data_path",
        default="data/text/bible_parallel_corpus.json",
        type=str,
        help="The path to the data file",
    )
    parser.add_argument(
        "--output_dir",
        default="data/encodings",
        type=str,
        help="Encodings output directory",
    )
    parser.add_argument(
        "--pruned_models_dir_prefix",
        type=str,
        default="artifacts/pruned",
        help="The prefix of the pruned models directory",
    )
    parser.add_argument(
        "--pruned_models_dir_postfix",
        type=str,
        default="mlm_finetuned",
        help="The prefix of the pruned models directory",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="*",
        required=False,
        choices=WIKIPEDIA_DUMPS.keys(),
        default=["en", "de", "es", "hi", "ru", "zh", "ar"],
        help=("The languages of subnetworks to use for encoding"),
    )
    parser.add_argument(
        "--use_base",
        action=argparse.BooleanOptionalAction,
        help="Use the base model for encoding",
    )
    parser.add_argument(
        "--use_mlm_finetuned",
        action=argparse.BooleanOptionalAction,
        help="Use the MLM finetuned model for encoding",
    )
    parser.add_argument(
        "--use_random_model",
        action=argparse.BooleanOptionalAction,
        help="Use a random model for encoding",
    )
    args = parser.parse_args()

    print(args)

    # Load language specific subnetworks
    pairs = [
        (
            f"{args.pruned_models_dir_prefix}_{lang}_{args.pruned_models_dir_postfix}",
            experiment_map[lang],
        )
        for lang in args.languages
    ]

    # Add base and MLM finetuned models
    if args.use_base:
        pairs.append(("FacebookAI/xlm-roberta-base", Experiments.XLMR_BASE))
    if args.use_mlm_finetuned:
        pairs.append(("artifacts/xlmr-mlm-finetuned", Experiments.XLMR_MLM_FINETUNED))
    if args.use_random_model:
        pairs.append(("FacebookAI/xlm-roberta-base", Experiments.XLMR_RANDOM))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    dataloaders, languages = get_bible_dataloaders_by_language(
        max_sentences=args.max_sentences,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        path=args.data_path,
    )

    for checkpoint, experiment in pairs:
        print("Loading model...")

        if args.use_random_model:
            print("Initializing the model with random weights")
            config = XLMRobertaConfig.from_pretrained(checkpoint)
            model = XLMRobertaModel(config)
        else:
            model = AutoModelForMaskedLM.from_pretrained(checkpoint)

        for language in languages:
            print(f"Encoding with a === {language} === subnetwork")
            encode(
                dataloader=dataloaders[language],
                model=model,
                language=language,
                experiment_name=experiment,
                output_dir=args.output_dir,
            )
