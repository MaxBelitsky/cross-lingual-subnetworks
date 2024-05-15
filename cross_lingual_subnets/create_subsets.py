import argparse
import os

from datasets import DatasetDict, load_dataset

WIKIPEDIA_DUMPS = {
    "lt": "20231101.lt",
    "ru": "20231101.ru",
    "cs": "20231101.cs",
    "en": "20231101.en",
    "fr": "20231101.fr",
    "de": "20231101.de",
    "zh": "20231101.zh",
    "sw": "20231101.sw",
    "ar": "20231101.ar",
    "hi": "20231101.hi",
    "fa": "20231101.fa",
    "ur": "20231101.ur",
    "es": "20231101.es",
}


def create_wikipedia_subsets(
    dumps,
    num_proc,
    seed,
    n_examples_per_lang,
    save_dir,
    push_to_hub=False,
    save_to_disk=False,
    save_separate_lang_datasets=False,
    hf_repo_id=None,
):
    """
    Create a subset of the wikipedia dataset.

    Args:
        dumps (dict): A dictionary with the wikipedia dumps for each language.
        num_proc (int): The number of processes to use.
        seed (int): The random seed.
        n_examples_per_lang (int): The number of examples per language to keep.
        save_dir (str): The directory to save the dataset.
        push_to_hub (bool): Whether to push the dataset to the hub.
        save_to_disk (bool): Whether to save the dataset to disk.
        save_separate_lang_datasets (bool): Whether to save the dataset for each language separately.
        hf_repo_id (str): The repository id.

    Returns:
        datasets.DatasetDict: The wikipedia dataset.
    """
    dataset_dict = {}
    for lang, dump in dumps.items():
        print(f"Loading {lang} wikipedia dataset")
        # Load the dataset
        dataset = load_dataset(
            "wikimedia/wikipedia", dump, num_proc=num_proc, split="train"
        )
        # Only keep text column
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col != "text"]
        )
        # Shuffle and select n_examples_per_lang
        if n_examples_per_lang < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(n_examples_per_lang))
        else:
            dataset = dataset.shuffle(seed=seed)

        if save_separate_lang_datasets:
            dataset.save_to_disk(
                f"{save_dir}/wikipedia_{lang}_{int(n_examples_per_lang/1000)}k"
            )
        dataset_dict[lang] = dataset

    final_dataset = DatasetDict(dataset_dict)

    if push_to_hub and hf_repo_id:
        final_dataset.push_to_hub(hf_repo_id)

    if save_to_disk:
        final_dataset.save_to_disk(f"{save_dir}/wikipedia_subset")

    return final_dataset


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_proc", type=int, default=os.cpu_count())
    argparser.add_argument("--n_examples_per_lang", type=int, default=80_000)
    argparser.add_argument("--save_separate_lang_datasets", action="store_true")
    argparser.add_argument("--save_to_disk", action=argparse.BooleanOptionalAction)
    argparser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction)
    argparser.add_argument("--save_dir", type=str, default="data")
    argparser.add_argument("--hf_repo_id", type=str, required=False)
    argparser.add_argument("--seed", type=int, default=42)
    
    args = argparser.parse_args()

    dataset = create_wikipedia_subsets(
        WIKIPEDIA_DUMPS,
        args.num_proc,
        n_examples_per_lang=args.n_examples_per_lang,
        save_dir=args.save_dir,
        push_to_hub=args.push_to_hub,
        save_to_disk=args.save_to_disk,
        save_separate_lang_datasets=args.save_separate_lang_datasets,
        hf_repo_id=args.hf_repo_id,
        seed=args.seed,
    )

    print(dataset)
