from transformers import AutoTokenizer
from config import CFG
from custom_model import CustomModel
from tqdm import tqdm
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch
import gc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained(CFG.backbone_model)
if CFG.is_replace:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[PARAGRAPH]"]}
    )
    print(len(tokenizer))

model = CustomModel(tokenizer=tokenizer)
print(model)

df = pd.read_csv("../dataset/test.csv")
if CFG.is_replace:
    df["full_text"] = df["full_text"].str.replace(
        r"\n\n", "[PARAGRAPH]", regex=True
    )


def tokenize(sample):
    return tokenizer(
        sample["full_text"],
        max_length=CFG.max_length,
        truncation=True,
        padding="max_length"
    )


ds = Dataset.from_pandas(df)
ds = ds.map(tokenize).remove_columns(["essay_id", "full_text"])


class DataCollator:
    def __call__(self, features):
        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"]
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


collator = DataCollator()
args = TrainingArguments(
    ".",
    per_device_eval_batch_size=1,
    report_to="none",
    remove_unused_columns=False
)

final_predictions = np.zeros((len(df), 1))

model_paths = []
for model_path in model_paths:
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda")
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        tokenizer=tokenizer
    )
    predictions = trainer.predict(ds).predictions
    final_predictions = final_predictions + predictions
    print(predictions)

del model
torch.cuda.empty_cache()
gc.collect()

final_predictions = final_predictions / len(model_paths) * 1.0
final_predictions = final_predictions.clip(1, 6).round()
print(final_predictions)

submission = pd.read_csv("../dataset/sample_submission.csv")
submission["score"] = final_predictions
submission.to_csv("submission.csv", index=False)

submission = pd.read_csv("submission.csv")
print(submission.head())
