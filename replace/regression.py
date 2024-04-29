import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import argparse
import warnings
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from sklearn.metrics import cohen_kappa_score
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
warnings.filterwarnings("ignore")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.MSELoss(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train(args):
    MODEL_NAME = args.model_name
    MAX_LENGTH = args.max_length
    NUM_SPLIT = args.num_split
    FOLD_ID = args.fold_id
    ACCESS_TOKEN = args.access_token
    TRAIN_FILE = args.train_file
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    ACCUMULATION_STEPS = args.accumulation_steps
    WARMUP_RATIO = args.warmup_ratio
    WEIGHT_DECAY = args.weight_decay
    SAVE_TOTAL_LIMIT = args.save_total_limit
    EPOCHS = args.epochs
    STEPS = args.steps
    LR_SCHEDULER = args.lr_scheduler

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=ACCESS_TOKEN)
    print(tokenizer.padding_side, tokenizer.pad_token)
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.padding_side, tokenizer.pad_token)
    print(len(tokenizer))
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[PARAGRAPH]"]}
    )
    print(len(tokenizer))

    df = pd.read_csv(TRAIN_FILE)
    df["full_text"] = df["full_text"].str.replace(r'\n\n', "[PARAGRAPH]", regex=True)
    df["labels"] = df.score.map(lambda x: x)
    df["labels"] = df["labels"].astype(float)
    X = df[["essay_id", "full_text", "score"]]
    y = df[["labels"]]

    skf = StratifiedKFold(n_splits=NUM_SPLIT, random_state=3047, shuffle=True)

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)

    global ds_train
    global ds_eval

    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if fold_id == FOLD_ID:
            print(f"... Fold {fold_id} ...")
            X_train, X_eval = X.iloc[train_index], X.iloc[val_index]
            y_train, y_eval = y.iloc[train_index], y.iloc[val_index]

            df_train = pd.concat([X_train, y_train], axis=1)
            df_train.reset_index(drop=True, inplace=True)
            print(df_train["labels"].value_counts())

            df_eval = pd.concat([X_eval, y_eval], axis=1)
            df_eval.reset_index(drop=True, inplace=True)
            print(df_eval["labels"].value_counts())

            ds_train = Dataset.from_pandas(df_train)
            print(ds_train)
            ds_eval = Dataset.from_pandas(df_eval)
            print(ds_eval)

            ds_train = ds_train.map(tokenize).remove_columns(
                ["essay_id", "full_text", "score"]
            )
            ds_eval = ds_eval.map(tokenize).remove_columns(
                ["essay_id", "full_text", "score"]
            )

    config = AutoConfig.from_pretrained(MODEL_NAME, token=ACCESS_TOKEN)
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        token=ACCESS_TOKEN,
        trust_remote_code=True,
        config=config
    )
    model.resize_token_embeddings(len(tokenizer))
    print(model.config.pad_token_id)
    model.config.pad_token_id = model.config.eos_token_id
    print(model.config.pad_token_id)
    print(model)

    print(torch.cuda.is_bf16_supported())

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
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
                pad_to_multiple_of=16
            )
            return batch

    def compute_metrics(p):
        preds, labels = p
        score = cohen_kappa_score(
            labels,
            preds.clip(1, 6).round(),
            weights="quadratic"
        )
        return {"qwk": score}

    training_args = TrainingArguments(
        output_dir=f"output_{MODEL_NAME.split('/')[-1]}/Fold{FOLD_ID}",
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        optim="paged_adamw_8bit",
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_strategy="steps",
        save_steps=STEPS,
        logging_steps=STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        save_only_model=True,
        lr_scheduler_type=LR_SCHEDULER,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LLM For Sequence Classification")
    parser.add_argument("--train_file", default="../dataset/train.csv", type=str)
    parser.add_argument("--model_name", default="google/gemma-2b", type=str)
    parser.add_argument("--max_length", default=1536, type=int)
    parser.add_argument("--num_split", default=10, type=int)
    parser.add_argument("--fold_id", default=0, type=int)
    parser.add_argument("--access_token", default="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", type=str)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--accumulation_steps", default=4, type=int)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--save_total_limit", default=10, type=int)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    args = parser.parse_args()
    print(args)
    train(args)
