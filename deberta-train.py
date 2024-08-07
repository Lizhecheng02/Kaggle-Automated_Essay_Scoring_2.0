from sklearn.metrics import cohen_kappa_score
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup
)
import pandas as pd
import numpy as np
import torch
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")


def train_deberta(args):
    MODEL_NAME = args.model_name
    FOLD = args.fold
    MAX_LENGTH = args.max_length

    df_train = pd.read_csv(args.train_file_path)
    df_train["fold"] = df_train.essay_id.map(lambda x: int(x, base=16) % args.fold_num)
    df_train["labels"] = df_train.score.map(lambda x: x - 1)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)

    ds_train = Dataset.from_pandas(df_train[df_train.fold != FOLD].copy())
    ds_eval = Dataset.from_pandas(df_train[df_train.fold == FOLD].copy())

    ds_train = ds_train.map(tokenize).remove_columns(["essay_id", "full_text", "score", "fold", "__index_level_0__"])
    ds_eval = ds_eval.map(tokenize).remove_columns(["essay_id", "full_text", "score", "fold", "__index_level_0__"])

    class DataCollator:
        def __call__(self, features):
            model_inputs = [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["labels"]
                } for feature in features
            ]

            batch = tokenizer.pad(
                model_inputs,
                padding=True,
                max_length="max_length",
                return_tensors="pt",
                pad_to_multiple_of=16
            )
            return batch

    def compute_metrics(p):
        preds, labels = p
        score = cohen_kappa_score(
            labels,
            preds.argmax(-1),
            weights="quadratic"
        )
        return {"qwk": score}

    wandb.login(key="c465dd55c08ec111e077cf0454ba111b3a764a78")
    wandb.init(
        project="aes2.0-kfold",
        job_type="training",
        anonymous="allow"
    )

    train_args = TrainingArguments(
        output_dir=f"fold_{FOLD}",
        fp16=True,
        # learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="wandb",
        evaluation_strategy="steps",
        do_eval=True,
        eval_steps=args.steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        save_steps=args.steps,
        logging_steps=args.steps,
        # lr_scheduler_type="linear",
        metric_for_best_model="qwk",
        greater_is_better=True,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_only_model=True,
        neftune_noise_alpha=args.neftune_noise_alpha
    )

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}],
        lr=args.learning_rate
    )

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_train_epochs * int(len(ds_train) * 1.0 / gpu_count / args.per_device_train_batch_size / args.gradient_accumulation_steps) * args.warmup_ratio,
        num_training_steps=args.num_train_epochs * int(len(ds_train) * 1.0 / gpu_count / args.per_device_train_batch_size / args.gradient_accumulation_steps),
        power=args.power,
        lr_end=args.lr_end
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=DataCollator(),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Deberta-v3 For Sequence Classification Task")
    parser.add_argument("--train_file_path", default="dataset/train.csv", type=str)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large", type=str)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--per_device_train_batch_size", default=1, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--save_total_limit", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--power", default=1.5, type=float)
    parser.add_argument("--lr_end", default=1e-6, type=float)
    parser.add_argument("--neftune_noise_alpha", default=0.05, type=float)
    args = parser.parse_args()
    print(args)
    train_deberta(args)
