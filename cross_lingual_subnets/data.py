from datasets import load_dataset

from cross_lingual_subnets.constants import Datasets


def get_dataset(dataset_name, tokenizer, cahce_dir=None):
    # Load the dataset
    dataset = load_dataset(dataset_name, cahce_dir=cahce_dir)

    # TODO: Preprocess/filter/map columns based on the dataset
    # The preprocessing might be different for each dataset
    # TODO: group the sentences together for the MLM task
    if dataset_name == Datasets.EXAMPLE:
        dataset.pop("unsupervised")
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"], padding=True, truncation=True, return_tensors="pt"
            ),
            batched=True,
            remove_columns=["text", "label"]
        )
    else:
        pass

    return dataset
