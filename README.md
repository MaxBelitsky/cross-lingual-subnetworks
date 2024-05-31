# cross-lingual-subnets

We investigate to what extent computation is shared across languages in multilingual models. In particular, we prune language-specific subnetworks from a multilingual model and check how much overlap there is between subnetworks for different languages and how these subnetworks perform across other languages. We also measure the similarity of the hidden states of a multilingual model for parallel sentences in different languages. All of these experiments suggest that a substantial amount of computation is being shared across languages in the multilingual model we use (XLM-R).

## Setup
- Create python virtual environment: `python -m venv venv`
- Activate the environmentt: `. ./venv/bin/activate`
- Install the requirements: `pip install -r requirements.txt`
- Create a .env file with the following variables: `WANDB_PROJECT`, `WANDB_ENTITY`

## Overview of repo

`cross_lingual_subnets/` the main code used for experiments

`scripts/` various files for running experiments, producing results for the final paper. Detailed explanations in the next section.

`outputs/` various outputs from the experiments (images, pruning masks, etc.)

`slurm_jobs/` job files used to schedule expensive computations

## Steps of the project

1. **Fine-tune the model on languages of interest**. This should bring out the language-specific capabilities of the model which should make it easier to extract language-specific subnetworks. We used 100k Wikipedia articles for each language: English, Spanish, Russian, German, Chinese, Arabic, Urdu, and Hindi.

1.1. `scripts/language.ipynb` Investigate and pick languages to fine-tune on.

1.2. `scripts/create_subsets.py` Get the Wikipedia data for languages.

1.3. `scripts/train.py` Fine-tune the model on the data.

2. **Identify language-specific subnetworks**. We attempt to extract language-specific subnetworks of a bigger network to investigate cross-lingual sharing mechanisms of deep neural networks.

2.1. `scripts/get_head_masks.py` Perform iterative structured pruning to extract language-specific subnetworks. We prune only attention heads, not MLP layers.

2.2. `scripts/prune.py` Load/Save pruned networks based on the pruning mask.

2.3. `scripts/proportion_masked.py` Compute the proportion of parameters masked per layer.

2.4. `scripts/mlm_plots.py` Compute the performance of all subnetworks on all other languages.

3. **Analyze subnetwork representations**. How similar are the subnetworks to each other? In terms of absolute overlap and representation similarity. How similar are the full model's and subnetworks' representations? How performative are the subnetworks on other languages?

TODO: add the bible parallel corpus on a file sharing website? HuggingFace?

3.1. `scripts/encode.py` Use the existing models to encode a parallel corpus.

3.2. `scripts/cka.ipynb` Use the encodings/representations and compare similarity (compute centered kernel alignment scores) for full and subnetwork models.

3.2. `scripts/rsa.ipynb` TODO

3.3. `scripts/isomorphic.ipynb` TODO

3.4. `scripts/token_overlap.py` Compute how many tokens of the other considered languages overlap with English showing potential bias of results.

3.5. `scripts/language.ipynb` Compute syntactic distance between the considered languages. The same similarity trend is visible in plots of `scripts/cka.ipynb`

## MLM Fine-tuning

To start training with the default arguments:
```
python -m scripts.train --dataset_name mbelitsky/wikipedia_subset --languages en es ru de zh ar hi ur
```

Results:
| Language | Perplexity before fine-tuning | Perplexity after fine-tuning |
|----------|:-----------------------------:|------------------------------|
| en       | 5.65                          | 4.15                         |
| es       | 5.07                          | 3.75                         |
| ru       | 3.80                          | 3.00                         |
| de       | 5.30                          | 3.93                         |
| zh       | 10.46                         | 6.73                         |
| ar       | 7.05                          | 4.29                         |
| ur       | 7.39                          | 4.50                         |
| hi       | 6.84                          | 4.50                         |

# Additional info
## MLM Data Preparation strategy
- N languages selected
- Wikipedia subsets of 80,000 articles for each language are created
- The articles are split and/or concatenated into chunks equal to the model context window size
- The training dataset is the shuffeled concatenation of all language subsets
- The test set is split by language (the default number of test examples for each language is set to 3000)



# Running the scripts
### Encoding
This script encodes sentences using language models and saves the encodings.
```
python -m scripts.encode --output_dir data/encodings --max_sentences 5000
```

Arguments:
| Argument                      | Type  | Required | Default Value                          | Description                                                                                  |
|-------------------------------|-------|----------|----------------------------------------|----------------------------------------------------------------------------------------------|
| `--batch_size`                | `int` | No       | 32                                     | The batch size of the dataloader.                                                            |
| `--max_sentences`             | `int` | No       | 1000                                   | The amount of sentences to encode.                                                           |
| `--data_path`                 | `str` | No       | `data/text/bible_parallel_corpus.json` | The path to the data file.                                                                   |
| `--output_dir`                | `str` | No       | `data/encodings`                       | Encodings output directory.                                                                  |
| `--pruned_models_dir_prefix`  | `str` | No       | `artifacts/pruned`                     | The prefix of the pruned models directory.                                                   |
| `--pruned_models_dir_postfix` | `str` | No       | `mlm_finetuned`                        | The postfix of the pruned models directory.                                                  |
| `--languages`                 | `str` | No       | `["en", "de", "es", "hi", "ru", "zh", "ar"]` | The languages of subnetworks to use for encoding. The options should be keys of `WIKIPEDIA_DUMPS`. |
| `--use_base`                  |       | No       |                                        | Use the base model for encoding.                                                             |
| `--use_mlm_finetuned`         |       | No       |                                        | Use the MLM finetuned model for encoding.                                                    |

### Pruning
This script generates a pruned model based on a specified head mask.

```
python -m scripts.prune --model_path models/fine-tuned-model --mask_dir fixed_prune_output --output_dir artifacts/pruned --languages ar de en es hi ru zh
```

Arguments:

| Argument      | Type  | Required | Description                                                                                  |
|----------------|-------|----------|----------------------------------------------------------------------------------------------|
| `--model_path` | `str` | Yes      | Path to the checkpoint for pruning. This should point to the model file that you want to prune. |
| `--mask_dir`   | `str` | Yes      | Path to the head mask to use. This should be a directory containing the mask that specifies which heads to prune. |
| `--output_dir` | `str` | Yes      | Path to save the pruned model. This is where the pruned model will be stored after the pruning process is completed. |
| `--languages`  | `str` | Yes      | The languages for creating subnetworks. This argument specifies the languages for which the subnetworks are to be created. The options should be keys of `WIKIPEDIA_DUMPS`. |


### Training

```
python -m scripts.train --dataset_name mbelitsky/wikipedia_subset --languages en es ru de zh ar hi ur
```

Arguments:
| Argument                      | Type    | Required | Default Value                          | Description                                                                                 |
|-------------------------------|---------|----------|----------------------------------------|---------------------------------------------------------------------------------------------|
| `--dataset_name`              | `str`   | Yes      |                                        | The datasets to train on. Choices are values from `Datasets.values()`.                      |
| `--model_checkpoint`          | `str`   | No       | `FacebookAI/xlm-roberta-base`          | The model variant to train.                                                                 |
| `--output_dir`                | `str`   | No       | `models/`                              | The output directory for the model.                                                         |
| `--seed`                      | `int`   | No       | 1234                                   | The random seed for reproducibility.                                                        |
| `--device`                    | `str`   | No       | None                                   | The device to use for training.                                                             |
| `--batch_size`                | `int`   | No       | 8                                      | The batch size.                                                                             |
| `--logging_steps`             | `int`   | No       | None                                   | The number of steps before logging metrics.                                                 |
| `--save_steps`                | `int`   | No       | None                                   | The number of steps before saving checkpoints.                                              |
| `--epochs`                    | `int`   | No       | 1                                      | The number of epochs.                                                                       |
| `--use_mps`                   |         | No       |                                        | Indicates whether to use MPS device.                                                        |
| `--use_fp16`                  |         | No       |                                        | Indicates whether to use mixed precision during training.                                   |
| `--eval_only`                 |         | No       |                                        | Indicates whether to skip training and only evaluate.                                       |
| `--lr`                        | `float` | No       | 2e-5                                   | The learning rate.                                                                          |
| `--weight_decay`              | `float` | No       | 0.01                                   | The weight decay.                                                                           |
| `--examples_per_lang`         | `int`   | No       | 100,000                                | The number of examples to use per language.                                                 |
| `--test_examples_per_lang`    | `int`   | No       | 3,000                                  | The number of examples to use for testing per language.                                     |
| `--languages`                 | `str`   | No       | All available in `WIKIPEDIA_DUMPS.keys()` | The languages to include in the dataset. If not provided, all languages are included.        |
| `--cache_dir`                 | `str`   | No       | None                                   | The cache directory for the dataset.                                                        |
| `--resume_from_checkpoint`    | `str`   | No       | None                                   | The checkpoint to resume training from.                                                     |
| `--last_run_id`               | `str`   | No       | None                                   | The wandb run ID to resume training from.                                                   |
| `--report_to`                 | `str`   | No       | `wandb`                                | The reporting tool. Default is 'wandb'. Set 'none' to disable reporting.                    |
