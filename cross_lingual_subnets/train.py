import argparse
import logging

from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from cross_lingual_subnets.utils import set_device, set_seed
from cross_lingual_subnets.data import get_dataset
from cross_lingual_subnets.constants import Datasets

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training sentence embedding models")

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The datasets to train on",
        required=True,
        choices=Datasets.values()
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="The model variant to train",
        default="FacebookAI/xlm-roberta-base"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="The output directory for the model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="The random seed for reproducibility"
        )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to use for training"
        )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="The batch size"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=False,
        help="The number of steps before logging metrics"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="The number of epochs"
    )
    parser.add_argument(
        "--use_mps",
        action=argparse.BooleanOptionalAction,
        help="Indicates whether to use MPS device"
    )
    parser.add_argument(
        "--use_fp16",
        action=argparse.BooleanOptionalAction,
        help="Indicates whether to use mixed precision during training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="The learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="The weight decay"
    )

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    args.device = args.device or set_device()

    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # Load the dataset and initialize collator
    dataset = get_dataset(args.dataset_name, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # Set up training arguments and the trainer
    logging_steps = args.logging_steps or len(dataset["train"]) // args.batch_size
    model_name = args.model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}{model_name}-finetuned",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=False,
        fp16=args.use_fp16,
        logging_steps=logging_steps,
        report_to="none", #TODO: set up WANDB logging
        use_mps_device=args.use_mps,
        data_seed=args.seed,
        seed=args.seed,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    trainer.save_model(args.output_dir)

    # Evaluate the model
    results = trainer.evaluate()
    print(results)
