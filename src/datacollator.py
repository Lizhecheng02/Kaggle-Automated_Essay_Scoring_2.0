from config import CFG
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained(CFG.backbone_model)
if CFG.is_replace:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[PARAGRAPH]"]}
    )
print("The Length of Tokenizer For DataCollator Is:", len(tokenizer))


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
