from transformers import AutoModelForMaskedLM, AutoTokenizer
from cross_lingual_subnets.constants import Experiments
import json
import torch


pairs = [
    ("FacebookAI/xlm-roberta-base", Experiments.XLMR_BASE),
    ("artifacts/checkpoint-h0nufx7r:v15", Experiments.XLMR_MLM_FINETUNED)
]

for checkpoint, experiment in pairs:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    with open("data/text/parallel-sentences.json", "r") as f:
        texts = json.load(f)

    en_texts = [txt["en"] for txt in texts.values()]
    ur_texts = [txt["ur"] for txt in texts.values()]
    hi_texts = [txt["hi"] for txt in texts.values()]


    def encode(texts: list, model, language="en", experiment_name=Experiments.XLMR_BASE):
        encoded_input = tokenizer(texts, return_tensors='pt', padding=True)

        # forward pass
        print("Predicting...")
        output = model(**encoded_input, output_hidden_states=True)
        # One for the output of the embeddings, if the model has an embedding layer, + one
        # for the output of each layer) of shape (batch_size, sequence_length, hidden_size)

        # TODO: is the outputs of the initial embedding layer added at the front or the back?
        # See https://huggingface.co/docs/transformers/v4.40.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel
        exp_destination = f"data/encodings/{language}_{experiment_name}.pt"
        print(f"Saving encodings to {exp_destination}")
        torch.save(output["hidden_states"][1:], exp_destination)


    encode(en_texts, model, language="en", experiment_name=experiment)
    encode(ur_texts, model, language="ur", experiment_name=experiment)
    encode(hi_texts, model, language="hi", experiment_name=experiment)
