from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from datasets import Dataset
from config import CFG
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def only4k(is_replace, num_split, selected_fold_id, backbone_model, max_length):
    if CFG.use_rubric:
        df = pd.read_csv("../dataset/4k_nooverlap_rubric.csv")
    else:
        df = pd.read_csv("../dataset/4k_nooverlap.csv")
    tokenizer = AutoTokenizer.from_pretrained(backbone_model)

    if is_replace:
        df["full_text"] = df["full_text"].str.replace(
            r'\n\n', "[PARAGRAPH]", regex=True
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[PARAGRAPH]"]}
        )

    df["labels"] = df.score.map(lambda x: x)
    df["labels"] = df["labels"].astype(float)
    X = df[["essay_id", "full_text", "score"]]
    y = df[["labels"]]

    skf = StratifiedKFold(n_splits=num_split, random_state=3047, shuffle=True)
    print(len(tokenizer))

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=max_length, truncation=True)

    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if fold_id == selected_fold_id:
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

            return tokenizer, ds_train, ds_eval


def only13k(is_replace, num_split, selected_fold_id, backbone_model, max_length):
    if CFG.use_rubric:
        df = pd.read_csv("../dataset/13k_overlap_rubric.csv")
    else:
        df = pd.read_csv("../dataset/13k_overlap.csv")
    tokenizer = AutoTokenizer.from_pretrained(backbone_model)

    if is_replace:
        df["full_text"] = df["full_text"].str.replace(
            r'\n\n', "[PARAGRAPH]", regex=True
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[PARAGRAPH]"]}
        )

    df["labels"] = df.score.map(lambda x: x)
    df["labels"] = df["labels"].astype(float)
    X = df[["essay_id", "full_text", "score"]]
    y = df[["labels"]]

    skf = StratifiedKFold(n_splits=num_split, random_state=3047, shuffle=True)
    print(len(tokenizer))

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=max_length, truncation=True)

    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if fold_id == selected_fold_id:
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

            return tokenizer, ds_train, ds_eval


def only17k(is_replace, num_split, selected_fold_id, backbone_model, max_length):
    if CFG.use_rubric:
        df = pd.read_csv("../dataset/train_rubric.csv")
    else:
        df = pd.read_csv("../dataset/train.csv")
    tokenizer = AutoTokenizer.from_pretrained(backbone_model)

    if is_replace:
        df["full_text"] = df["full_text"].str.replace(
            r'\n\n', "[PARAGRAPH]", regex=True
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[PARAGRAPH]"]}
        )

    df["labels"] = df.score.map(lambda x: x)
    df["labels"] = df["labels"].astype(float)
    X = df[["essay_id", "full_text", "score"]]
    y = df[["labels"]]

    skf = StratifiedKFold(n_splits=num_split, random_state=3047, shuffle=True)
    print(len(tokenizer))

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=max_length, truncation=True)

    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if fold_id == selected_fold_id:
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

            return tokenizer, ds_train, ds_eval
        

def only19k(is_replace, num_split, selected_fold_id, backbone_model, max_length):
    if CFG.use_rubric:
        df = pd.read_csv("../dataset/19k_same_distribution_rubric.csv")
    else:
        df = pd.read_csv("../dataset/19k_same_distribution.csv")
    tokenizer = AutoTokenizer.from_pretrained(backbone_model)

    if is_replace:
        df["full_text"] = df["full_text"].str.replace(
            r'\n\n', "[PARAGRAPH]", regex=True
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[PARAGRAPH]"]}
        )

    df["labels"] = df.score.map(lambda x: x)
    df["labels"] = df["labels"].astype(float)
    X = df[["essay_id", "full_text", "score"]]
    y = df[["labels"]]

    skf = StratifiedKFold(n_splits=num_split, random_state=3047, shuffle=True)
    print(len(tokenizer))

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=max_length, truncation=True)

    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if fold_id == selected_fold_id:
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

            return tokenizer, ds_train, ds_eval


def only30k(is_replace, num_split, selected_fold_id, backbone_model, max_length):
    if CFG.use_rubric:
        df = pd.read_csv("../dataset/30k_train_rubric.csv")
    else:
        df = pd.read_csv("../dataset/30k_train.csv")
    tokenizer = AutoTokenizer.from_pretrained(backbone_model)

    if is_replace:
        df["full_text"] = df["full_text"].str.replace(
            r'\n\n', "[PARAGRAPH]", regex=True
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[PARAGRAPH]"]}
        )

    df["labels"] = df.score.map(lambda x: x)
    df["labels"] = df["labels"].astype(float)
    X = df[["essay_id", "full_text", "score"]]
    y = df[["labels"]]

    skf = StratifiedKFold(n_splits=num_split, random_state=3047, shuffle=True)
    print(len(tokenizer))

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=max_length, truncation=True)

    for fold_id, (train_index, val_index) in enumerate(skf.split(X, y)):
        if fold_id == selected_fold_id:
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

            return tokenizer, ds_train, ds_eval


def train17k_validation4k(is_replace, num_split, selected_fold_id, backbone_model, max_length):
    if CFG.use_rubric:
        df_13k = pd.read_csv("../dataset/13k_overlap_rubric.csv")
        df_4k = pd.read_csv("../dataset/4k_nooverlap_rubric.csv")
    else:
        df_13k = pd.read_csv("../dataset/13k_overlap.csv")
        df_4k = pd.read_csv("../dataset/4k_nooverlap.csv")
    tokenizer = AutoTokenizer.from_pretrained(backbone_model)

    if is_replace:
        df_13k["full_text"] = df_13k["full_text"].str.replace(
            r'\n\n', "[PARAGRAPH]", regex=True
        )
        df_4k["full_text"] = df_4k["full_text"].str.replace(
            r'\n\n', "[PARAGRAPH]", regex=True
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[PARAGRAPH]"]}
        )

    df_13k["labels"] = df_13k.score.map(lambda x: x)
    df_13k["labels"] = df_13k["labels"].astype(float)

    df_4k["labels"] = df_4k.score.map(lambda x: x)
    df_4k["labels"] = df_4k["labels"].astype(float)
    X_4k = df_4k[["essay_id", "full_text", "score"]]
    y_4k = df_4k[["labels"]]

    skf = StratifiedKFold(n_splits=num_split, random_state=3047, shuffle=True)
    print(len(tokenizer))

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=max_length, truncation=True)

    for fold_id, (train_index, val_index) in enumerate(skf.split(X_4k, y_4k)):
        if fold_id == selected_fold_id:
            print(f"... Fold {fold_id} ...")
            X_train, X_eval = X_4k.iloc[train_index], X_4k.iloc[val_index]
            y_train, y_eval = y_4k.iloc[train_index], y_4k.iloc[val_index]

            df_train = pd.concat([X_train, y_train], axis=1)
            df_train = pd.concat([df_train, df_13k], axis=0)
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

            return tokenizer, ds_train, ds_eval


def get_tokenizer_and_dataset():
    if CFG.validation_type == "only4k":
        tokenizer, ds_train, ds_eval = only4k(
            is_replace=CFG.is_replace,
            num_split=CFG.num_split,
            selected_fold_id=CFG.selected_fold_id,
            backbone_model=CFG.backbone_model,
            max_length=CFG.max_length
        )

        return tokenizer, ds_train, ds_eval

    elif CFG.validation_type == "only13k":
        tokenizer, ds_train, ds_eval = only13k(
            is_replace=CFG.is_replace,
            num_split=CFG.num_split,
            selected_fold_id=CFG.selected_fold_id,
            backbone_model=CFG.backbone_model,
            max_length=CFG.max_length
        )

        return tokenizer, ds_train, ds_eval

    elif CFG.validation_type == "only17k":
        tokenizer, ds_train, ds_eval = only17k(
            is_replace=CFG.is_replace,
            num_split=CFG.num_split,
            selected_fold_id=CFG.selected_fold_id,
            backbone_model=CFG.backbone_model,
            max_length=CFG.max_length
        )

        return tokenizer, ds_train, ds_eval
    
    elif CFG.validation_type == "only19k":
        tokenizer, ds_train, ds_eval = only19k(
            is_replace=CFG.is_replace,
            num_split=CFG.num_split,
            selected_fold_id=CFG.selected_fold_id,
            backbone_model=CFG.backbone_model,
            max_length=CFG.max_length
        )

        return tokenizer, ds_train, ds_eval

    elif CFG.validation_type == "only30k":
        tokenizer, ds_train, ds_eval = only30k(
            is_replace=CFG.is_replace,
            num_split=CFG.num_split,
            selected_fold_id=CFG.selected_fold_id,
            backbone_model=CFG.backbone_model,
            max_length=CFG.max_length
        )

        return tokenizer, ds_train, ds_eval

    elif CFG.validation_type == "train17k_validation4k":
        tokenizer, ds_train, ds_eval = train17k_validation4k(
            is_replace=CFG.is_replace,
            num_split=CFG.num_split,
            selected_fold_id=CFG.selected_fold_id,
            backbone_model=CFG.backbone_model,
            max_length=CFG.max_length
        )

        return tokenizer, ds_train, ds_eval
