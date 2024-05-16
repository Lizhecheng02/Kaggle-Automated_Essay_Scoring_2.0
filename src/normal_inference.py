from transformers import AutoTokenizer
from config import CFG
from custom_model import CustomModel
from tqdm import tqdm
import pandas as pd
import torch
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

full_texts = df["full_text"].tolist()
encoded_inputs = tokenizer(
    full_texts,
    max_length=CFG.max_length,
    truncation=True,
    padding=True,
    return_tensors="pt"
)
encoded_inputs = encoded_inputs.to("cuda")

final_predictions = []
all_model_predictions = [[] for _ in range(len(df))]

model_paths = []
for model_path in model_paths:
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda")

    with torch.no_grad():
        for idx in tqdm(range(len(df))):
            batch = {k: v[idx:idx + 1] for k, v in encoded_inputs.items()}
            prediction = model(batch).logits.cpu().detach().numpy()[0][0]
            print(prediction)
            all_model_predictions[idx].append(prediction)

for sample_predictions in all_model_predictions:
    final_prediction = sum(sample_predictions) / len(sample_predictions)
    final_prediction = round(np.clip(final_prediction, 1, 6))
    final_predictions.append(final_prediction)

print(final_predictions)

submission = pd.read_csv("../dataset/sample_submission.csv")
submission["score"] = final_predictions
submission.to_csv("submission.csv", index=False)

submission = pd.read_csv("submission.csv")
print(submission.head())
