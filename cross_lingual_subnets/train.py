import os
import argparse
import logging

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from dotenv import load_dotenv
import wandb

from cross_lingual_subnets.utils import set_device, set_seed
from cross_lingual_subnets.data import get_dataset
from cross_lingual_subnets.constants import Datasets
from cross_lingual_subnets.trainer import CustomTrainer
from cross_lingual_subnets.create_subsets import WIKIPEDIA_DUMPS

logger = logging.getLogger(__name__)

load_dotenv()
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training sentence embedding models")

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The datasets to train on",
        required=True,
        choices=Datasets.values(),
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="The model variant to train",
        default="FacebookAI/xlm-roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="The output directory for the model",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="The random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="The device to use for training"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size")
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=False,
        help="The number of steps before logging metrics",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        required=False,
        help="The number of steps before logging metrics",
    )
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs")
    parser.add_argument(
        "--use_mps",
        action=argparse.BooleanOptionalAction,
        help="Indicates whether to use MPS device",
    )
    parser.add_argument(
        "--use_fp16",
        action=argparse.BooleanOptionalAction,
        help="Indicates whether to use mixed precision during training",
    )
    parser.add_argument(
        "--eval_only",
        action=argparse.BooleanOptionalAction,
        help="Indicates whether to skip training",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="The weight decay"
    )
    parser.add_argument(
        "--examples_per_lang",
        type=int,
        default=100_000,
        help="The number of examples to use for per language",
    )
    parser.add_argument(
        "--test_examples_per_lang",
        type=int,
        default=3000,
        help="The number of examples to use for testing per language",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="*",
        required=False,
        choices=WIKIPEDIA_DUMPS.keys(),
        help="The languages to include in the dataset. If not provided, all languages are included.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache directory for the dataset",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="The checkpoint to resume training from",
    )
    parser.add_argument(
        "--last_run_id",
        type=str,
        default=None,
        help="The wandb run id to resume training from",
    )

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    args.device = args.device or set_device()

    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # Load the dataset and initialize collator
    dataset = get_dataset(
        args.dataset_name,
        tokenizer,
        seed=args.seed,
        test_size=args.test_examples_per_lang,
        n_examples_per_lang=args.examples_per_lang,
        cache_dir=args.cache_dir,
        languages=args.languages,
    )
    # TODO: this collator masks tokens randomly so we can get different values on two different evaluations
    # maybe it is needed to mask the tokens once before training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # Set up training arguments and the trainer
    logging_steps = args.logging_steps or len(dataset["train"]) // args.batch_size
    save_steps = args.save_steps or logging_steps
    model_name = args.model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}{model_name}-finetuned",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=False,
        fp16=args.use_fp16,
        logging_steps=logging_steps,
        eval_steps=logging_steps,
        save_steps=save_steps,
        report_to="wandb",
        use_mps_device=args.use_mps,
        data_seed=args.seed,
        seed=args.seed,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    if not args.eval_only:
        if args.last_run_id:
            with wandb.init(
                project=os.environ["WANDB_PROJECT"],
                id=args.last_run_id,
                resume="must",
            ) as run:
                trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
                trainer.save_model(args.output_dir)
        else:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            trainer.save_model(args.output_dir)

    # Evaluate the model
    results = trainer.evaluate()
    print(results)
