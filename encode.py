from transformers import AutoModelForMaskedLM, AutoTokenizer
from cross_lingual_subnets.constants import Experiments
import json
import os
import torch
from collections import defaultdict
import sys
from torch.utils.data import DataLoader


os.environ["TOKENIZERS_PARALLELISM"] = "false"
BATCH_SIZE = 32
MAX_SENTENCES = 1000


def encode(
    dataloader: DataLoader,
    model,
    tokenizer,
    language="en",
    experiment_name=Experiments.XLMR_BASE,
):
    # encoded_input = tokenizer(texts, return_tensors="pt", padding=True)
    # # print(encoded_input.batch_size)
    # # pipe = pipeline("masked-language-modeling", model=model, tokenizer=tokenizer, batch_size=32)
    # # encoded_input = encoded_input.reshape()
    # # text_chunks = chunks(texts, 100)

    # # output = pipe(texts)
    # # print("wooooooo")
    # sys.exit(1)

    # Save hidden outputs per layer
    hids = {}
    for i, batch in enumerate(dataloader):
        print(f"Predicting batch {i+1}/{len(dataloader)}")

        # One for the output of the embeddings, if the model has an embedding layer, + one
        # for the output of each layer of shape (batch_size, sequence_length, hidden_size)
        # TODO: is the outputs of the initial embedding layer added at the front or the back?
        # See https://huggingface.co/docs/transformers/v4.40.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel
        output = model(**batch, output_hidden_states=True)["hidden_states"][1:]
        # Aggregate averages of sentences (hence mean) per layer
        for j in range(len(output)):
            if i == 0:
                hids[j] = output[j].mean(dim=1)
            else:
                out = output[j].mean(dim=1)
                hids[j] = torch.cat((hids[j], out), 0)

    exp_destination = f"data/encodings/{experiment_name}/{language}.pt"
    print(f"Saving encodings to {exp_destination}")
    if not os.path.exists(f"data/encodings/{experiment_name}/"):
        os.makedirs(f"data/encodings/{experiment_name}/")

    print(hids.keys())
    print(hids[0].shape)
    torch.save(hids, exp_destination)


def collate_batch(batch, tokenizer):
    encoded_input = tokenizer(batch, return_tensors="pt", padding=True)
    return encoded_input


def get_bible_dataloaders_by_language(tokenizer) -> tuple[dict, list]:
    with open("data/text/bible_parallel_corpus.json", "r", encoding="ISO-8859-1") as f:
        texts = json.load(f)

    texts_by_language = defaultdict(list)
    languages = texts[0].keys()
    for text in texts:
        for lang in languages:
            texts_by_language[lang].append(text[lang])

    # Define a custom collate function which takes in a vocab
    collate_fn = lambda batch: collate_batch(batch, tokenizer=tokenizer)  # noqa

    data_loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }

    dataloaders = {
        language: DataLoader(
            texts_by_language[language][:MAX_SENTENCES], **data_loader_kwargs
        )
        for language in languages
    }

    return dataloaders, languages


if __name__ == "__main__":
    pairs = [
        ("FacebookAI/xlm-roberta-base", Experiments.XLMR_BASE),
        ("artifacts/xlmr-mlm-finetuned", Experiments.XLMR_MLM_FINETUNED),
        ("artifacts/pruned_ar_mlm_finetuned", Experiments.AR_SUB_MLM_FINETUNED),
        ("artifacts/pruned_de_mlm_finetuned", Experiments.DE_SUB_MLM_FINETUNED),
        ("artifacts/pruned_en_mlm_finetuned", Experiments.EN_SUB_MLM_FINETUNED),
        ("artifacts/pruned_es_mlm_finetuned", Experiments.ES_SUB_MLM_FINETUNED),
        ("artifacts/pruned_hi_mlm_finetuned", Experiments.HI_SUB_MLM_FINETUNED),
        ("artifacts/pruned_ru_mlm_finetuned", Experiments.RU_SUB_MLM_FINETUNED),
        ("artifacts/pruned_zh_mlm_finetuned", Experiments.ZH_SUB_MLM_FINETUNED),
    ]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    dataloaders, languages = get_bible_dataloaders_by_language(tokenizer)

    for checkpoint, experiment in pairs:
        print("Loading model...")
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)

        for language in languages:
            encode(
                dataloader=dataloaders[language],
                model=model,
                tokenizer=tokenizer,
                language=language,
                experiment_name=experiment,
            )
