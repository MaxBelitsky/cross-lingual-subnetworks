# Investigating the cross-lingual sharing mechanism of multilingual models through their subnetworks

## Setup
- Create python virtual environment: `python -m venv venv`
- Activate the environmentt: `. ./venv/bin/activate`
- Install the requirements: `pip install -r requirements.txt`
- Create a .env file with the following variables: `WANDB_PROJECT`, `WANDB_ENTITY`

## MLM Fine-tuning

To start training with the default arguments:
```
python -m cross_lingual_subnets.train --dataset_name mbelitsky/wikipedia_subset --languages en fr
```
