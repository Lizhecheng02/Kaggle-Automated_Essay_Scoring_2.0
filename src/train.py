from awp_custom_trainer import AWPCustomTrainer
from compute_metrics import compute_metrics
from custom_model import CustomModel, Get_AutoModel
from custom_trainer import CustomTrainer
# from datacollator import DataCollator
from dataset import get_tokenizer_and_dataset
from scheduler import get_scheduler
from training_arguments import get_training_arguments
from wandb_init import wandb_init
from config import CFG
from transformers import AdamW, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings("ignore")

tokenizer, ds_train, ds_eval = get_tokenizer_and_dataset()

if CFG.use_autoclassification:
    model = Get_AutoModel(tokenizer=tokenizer)
else:
    model = CustomModel(tokenizer=tokenizer)
print(model)

wandb_init()

training_args = get_training_arguments()
optimizer = AdamW(model.parameters(), lr=CFG.learning_rate)
scheduler = get_scheduler(
    ds_train=ds_train,
    gpu_count=torch.cuda.device_count(),
    optimizer=optimizer
)


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
            max_length=CFG.max_length,
            return_tensors="pt",
            pad_to_multiple_of=16
        )
        return batch


if CFG.train_type == "awp":
    trainer = AWPCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        awp_lr=CFG.awp_lr,
        awp_eps=CFG.awp_eps,
        awp_start_epoch=CFG.awp_start_epoch
    )

elif CFG.train_type == "no-awp":
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

else:
    raise ValueError(f"Invalid Training Type: {CFG.train_type}")

trainer.train()
