from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup
)
import pandas as pd
import torch
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")

class AWP:
    def __init__(self, model, adv_param="weight", adv_lr=0.1, adv_eps=1e-4):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (self.backup[name] - grad_eps, self.backup[name] + grad_eps)

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


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
        preprocess_logits_for_metrics=None,
        awp_lr=0.1,
        awp_eps=1e-4,
        awp_start_epoch=1.0
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

        self.awp_lr = awp_lr
        self.awp_eps = awp_eps
        self.awp_start_epoch = awp_start_epoch

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model"s documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        o_inputs = inputs.copy()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        ########################
        # AWP
        if self.awp_lr != 0 and self.state.epoch >= self.awp_start_epoch:
           # print(inputs)
           # print("Start amp")
            self.awp = AWP(model, adv_lr=self.awp_lr, adv_eps=self.awp_eps)
            self.awp._save()
            self.awp._attack_step()
            with self.compute_loss_context_manager():
                awp_loss = self.compute_loss(self.awp.model, o_inputs)

            if self.args.n_gpu > 1:
                awp_loss = awp_loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(awp_loss, self.optimizer) as awp_scaled_loss:
                    awp_scaled_loss.backward()
            else:
                self.accelerator.backward(awp_loss)
            self.awp._restore()
        ########################

        return loss.detach() / self.args.gradient_accumulation_steps


def train_deberta(args):
    MODEL_NAME = args.model_name
    MAX_LENGTH = args.max_length

    df_train = pd.read_csv(args.train_file_path)
    df_train["labels"] = df_train.score.map(lambda x: x - 1)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    X = df_train[["essay_id", "full_text", "score"]]
    y = df_train[["labels"]]

    skf = StratifiedKFold(n_splits=args.fold_num, random_state=args.random_state, shuffle=True)
    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if (fold_id + 1) == args.select_fold:
            print(f"... Fold {fold_id + 1} ...")
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

            def tokenize(sample):
                return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)

            ds_train = ds_train.map(tokenize).remove_columns(["essay_id", "full_text", "score"])
            ds_eval = ds_eval.map(tokenize).remove_columns(["essay_id", "full_text", "score"])

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
                    preds.argmax(-1),
                    weights="quadratic"
                )
                return {"qwk": score}

            wandb.login(key="c465dd55c08ec111e077cf0454ba111b3a764a78")
            wandb.init(
                project="awp-skf-train",
                job_type="training",
                anonymous="allow"
            )

            train_args = TrainingArguments(
                output_dir=f"output_{fold_id + 1}",
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

            trainer = CustomTrainer(
                model=model,
                args=args,
                train_dataset=ds_train,
                eval_dataset=ds_eval,
                data_collator=DataCollator(),
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                optimizers=(optimizer, scheduler),
                awp_lr=args.awp_lr,
                awp_eps=args.awp_eps,
                awp_start_epoch=args.awp_start_epoch
            )

            trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Deberta-v3 For Sequence Classification Task")
    parser.add_argument("--train_file_path", default="dataset/train.csv", type=str)
    parser.add_argument("--random_state", default=2024, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--select_fold", default=1, type=int)
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
    parser.add_argument("--awp_lr", default=0.1, type=float)
    parser.add_argument("--awp_eps", default=1e-4, type=float)
    parser.add_argument("--awp_start_epoch", default=1.0, type=float)
    args = parser.parse_args()
    print(args)
    train_deberta(args)
