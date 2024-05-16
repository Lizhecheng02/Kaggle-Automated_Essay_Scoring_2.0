from transformers import TrainingArguments
from config import CFG
from datetime import datetime
import torch
import warnings
warnings.filterwarnings("ignore")


def get_training_arguments():
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second

    time_string = f"{year}{month}{day}-{hour}:{minute}:{second}"

    return TrainingArguments(
        output_dir=f"output_{CFG.backbone_model.split('/')[-1]}_{time_string}/Fold{CFG.selected_fold_id}",
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        learning_rate=CFG.learning_rate,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        num_train_epochs=CFG.num_train_epochs,
        weight_decay=CFG.weight_decay,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=CFG.steps,
        save_strategy="steps",
        save_steps=CFG.steps,
        logging_steps=CFG.steps,
        save_total_limit=CFG.save_total_limit,
        metric_for_best_model="qwk",
        greater_is_better=True,
        save_only_model=True,
        report_to="wandb" if CFG.use_wandb else "none",
        remove_unused_columns=False,
        label_names=["labels"]
    )
