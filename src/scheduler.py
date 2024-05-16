from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from config import CFG
import warnings
warnings.filterwarnings("ignore")


def return_polynomial_decay_schedule_with_warmup(ds_train, gpu_count, optimizer):
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(CFG.num_train_epochs * len(ds_train) * 1.0 / gpu_count /
                             CFG.per_device_train_batch_size / CFG.gradient_accumulation_steps * CFG.warmup_ratio),
        num_training_steps=int(CFG.num_train_epochs * len(ds_train) * 1.0 /
                               gpu_count / CFG.per_device_train_batch_size / CFG.gradient_accumulation_steps),
        power=CFG.power,
        lr_end=CFG.lr_end
    )

    return scheduler


def return_cosine_schedule_with_warmup(ds_train, gpu_count, optimizer):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(CFG.num_train_epochs * len(ds_train) * 1.0 / gpu_count /
                             CFG.per_device_train_batch_size / CFG.gradient_accumulation_steps * CFG.warmup_ratio),
        num_training_steps=int(CFG.num_train_epochs * len(ds_train) * 1.0 /
                               gpu_count / CFG.per_device_train_batch_size / CFG.gradient_accumulation_steps)
    )

    return scheduler


def return_constant_schedule_with_warmup(ds_train, gpu_count, optimizer):
    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(CFG.num_train_epochs * len(ds_train) * 1.0 / gpu_count /
                             CFG.per_device_train_batch_size / CFG.gradient_accumulation_steps * CFG.warmup_ratio),
    )

    return scheduler


def return_linear_schedule_with_warmup(ds_train, gpu_count, optimizer):
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(CFG.num_train_epochs * len(ds_train) * 1.0 / gpu_count /
                             CFG.per_device_train_batch_size / CFG.gradient_accumulation_steps * CFG.warmup_ratio),
        num_training_steps=int(CFG.num_train_epochs * len(ds_train) * 1.0 /
                               gpu_count / CFG.per_device_train_batch_size / CFG.gradient_accumulation_steps)
    )

    return scheduler


def get_scheduler(ds_train, gpu_count, optimizer):
    if CFG.scheduler_type == "polynomial":
        return return_polynomial_decay_schedule_with_warmup(ds_train, gpu_count, optimizer)

    elif CFG.scheduler_type == "cosine":
        return return_cosine_schedule_with_warmup(ds_train, gpu_count, optimizer)

    elif CFG.scheduler_type == "constant":
        return return_constant_schedule_with_warmup(ds_train, gpu_count, optimizer)

    elif CFG.scheduler_type == "linear":
        return return_linear_schedule_with_warmup(ds_train, gpu_count, optimizer)

    else:
        raise ValueError(f"Invalid Scheduler Type: {CFG.scheduler_type}")
