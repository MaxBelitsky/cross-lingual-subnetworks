# cross-lingual-subnets

Investigating the cross-lingual sharing mechanism of multilingual models through their subnetworks.

TODO: add further explanation about the project (e.g. abstract)

TODO: add edited poster

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

# Additional info
## MLM Data Preparation strategy
- N languages selected
- Wikipedia subsets of 80,000 articles for each language are created
- The articles are split and/or concatenated into chunks equal to the model context window size
- The training dataset is the shuffeled concatenation of all language subsets
- The test set is split by language (the default number of test examples for each language is set to 3000)
