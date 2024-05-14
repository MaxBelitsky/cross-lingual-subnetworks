# Investigating the cross-lingual sharing mechanism of multilingual models through their subnetworks

## Setup
- Create python virtual environment: `python -m venv venv`
- Activate the environmentt: `. ./venv/bin/activate`
- Install the requirements: `pip install -r requirements.txt`
- Create a .env file with the following variables: `WANDB_PROJECT`, `WANDB_ENTITY`

## MLM Fine-tuning

To start training with the default arguments:
```
python -m cross_lingual_subnets.train --dataset_name mbelitsky/wikipedia_subset --languages en sp ru de zh ar hi ur
```

# Additional info
## MLM Data Preparation strategy
- N languages selected
- Wikipedia subsets of 80,000 articles for each language are created
- The articles are split and/or concatenated into chunks equal to the model context window size
- The training dataset is the shuffeled concatenation of all language subsets
- The test set is split by language (the default number of test examples for each language is set to 3000)
