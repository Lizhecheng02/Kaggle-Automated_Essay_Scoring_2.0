from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AdamW,
    get_polynomial_decay_schedule_with_warmup
)
import os
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")


def get_last_hidden_state(backbone_outputs):
    last_hidden_state = backbone_outputs[0]
    return last_hidden_state


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


def get_input_ids(inputs):
    return inputs["input_ids"]


def get_attention_mask(inputs):
    return inputs["attention_mask"]


def train(args):
    class AttentionPooling(nn.Module):
        def __init__(self, hiddendim_fc, dropout):
            super(AttentionPooling, self).__init__()
            self.num_hidden_layers = AutoConfig.from_pretrained(args.model_name).num_hidden_layers
            self.hidden_size = AutoConfig.from_pretrained(args.model_name).hidden_size
            self.hiddendim_fc = hiddendim_fc
            self.dropout = nn.Dropout(dropout)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
            self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
            w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
            self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)

            self.output_dim = self.hiddendim_fc

        def attention(self, h):
            v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
            v = F.softmax(v, -1)
            v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
            v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
            return v

        def forward(self, inputs, backbone_outputs):
            all_hidden_states = get_all_hidden_states(backbone_outputs)
            hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze() for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
            hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
            out = self.attention(hidden_states)
            out = self.dropout(out)
            return out


    class CustomModel(nn.Module):
        def __init__(self, tokenizer):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(args.model_name)
            print(self.backbone)
            self.backbone.resize_token_embeddings(len(tokenizer))
            self.pool = AttentionPooling(hiddendim_fc=args.hiddendim_fc, dropout=args.dropout)
            self.fc = nn.Linear(self.pool.output_dim, 1)

        def forward(self, inputs):
            outputs = self.backbone(**inputs, output_hidden_states=True)
            feature = self.pool(inputs, outputs)
            output = self.fc(feature)

            return SequenceClassifierOutputWithPast(
                loss=None,
                logits=output,
                past_key_values=None,
                hidden_states=None,
                attentions=None
            )

    class CustomTrainer(Trainer):
        def __init__(
            self,
            model=None,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None
        ):
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
            
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(inputs)
            logits = outputs.logits
            loss = nn.MSELoss()(logits.view(-1), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        def _save_checkpoint(self, model, trial, metrics=None):
            # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
            # want to save except FullyShardedDDP.
            # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"
    
            # Save model checkpoint
            PREFIX_CHECKPOINT_DIR = "checkpoint"
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
    
            if self.hp_search_backend is None and trial is None:
                self.store_flos()
    
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            # self.save_model(output_dir, _internal_call=True)  # 不要保存为HuggingFace的格式
            os.makedirs(output_dir)
            torch.save(model.state_dict(), f"{output_dir}/model-{self.state.global_step}.pth") # 保存为pth文件格式
            
    
            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)
    
            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]
    
                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir
    
            # Save the Trainer state
            # if self.args.should_save:
            #     # Update the `TrainerControl` state to where we are currently
            #     self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            #     self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
    
            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
    
            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    df = pd.read_csv(args.train_file)
    df["full_text"] = df["full_text"].str.replace(r'\n\n', "[PARAGRAPH]", regex=True)
    df["labels"] = df.score.map(lambda x: x)
    df["labels"] = df["labels"].astype(float)
    X = df[["essay_id", "full_text", "score"]]
    y = df[["labels"]]

    skf = StratifiedKFold(n_splits=args.num_split, random_state=3047, shuffle=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[PARAGRAPH]"]}
    )
    print(len(tokenizer))

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=args.max_length, truncation=True)

    global ds_train
    global ds_eval

    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if fold_id == args.fold_id:
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
            ds_eval = Dataset.from_pandas(df_eval)

            ds_train = ds_train.map(tokenize).remove_columns(
                ["essay_id", "full_text", "score"]
            )
            ds_eval = ds_eval.map(tokenize).remove_columns(
                ["essay_id", "full_text", "score"]
            )

    print(ds_train)
    print(ds_eval)

    model = CustomModel(tokenizer=tokenizer)
    print(model)

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
                max_length=args.max_length,
                return_tensors="pt",
                pad_to_multiple_of=16
            )
            return batch

    def compute_metrics(p):
        preds, labels = p
        # print(preds)
        # print(labels)
        score = cohen_kappa_score(
            labels,
            preds.clip(1, 6).round(),
            weights="quadratic"
        )
        return {"qwk": score}

    training_args = TrainingArguments(
        output_dir=f"output_{args.model_name.split('/')[-1]}/Fold{args.fold_id}",
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        save_steps=args.steps,
        logging_steps=args.steps,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"]
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.epochs * len(ds_train) * 1.0 / gpu_count / args.batch_size / args.accumulation_steps * args.warmup_ratio),
        num_training_steps=int(args.epochs * len(ds_train) * 1.0 / gpu_count / args.batch_size / args.accumulation_steps),
        power=args.power,
        lr_end=args.lr_end
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LLM For Sequence Classification")
    parser.add_argument("--train_file", default="../../dataset/train.csv", type=str)
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large", type=str)
    parser.add_argument("--hiddendim_fc", default=512, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--max_length", default=1536, type=int)
    parser.add_argument("--num_split", default=5, type=int)
    parser.add_argument("--fold_id", default=0, type=int)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--accumulation_steps", default=4, type=int)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--save_total_limit", default=5, type=int)
    parser.add_argument("--power", default=2.0, type=float)
    parser.add_argument("--lr_end", default=1.0e-6, type=float)
    args = parser.parse_args()
    print(args)
    train(args)