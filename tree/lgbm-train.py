import gc
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import log_evaluation, early_stopping
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm
from essay_processor import EssayProcessor
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold
warnings.filterwarnings("ignore")

train = pd.read_csv("../dataset/train.csv")
# train = train[:10000]
test = pd.read_csv("../dataset/test.csv")
# test = train[10000:12000]
# test.drop(columns=["score"], axis=1, inplace=True)


class CFG:
    random_state = 2024
    LOWER_CASE = False
    VOCAB_SIZE = 32000


# -------- 训练tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + [normalizers.Lowercase()] if CFG.LOWER_CASE else []
)
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(
    vocab_size=CFG.VOCAB_SIZE,
    special_tokens=special_tokens
)

dataset = Dataset.from_pandas(test[["full_text"]])


def train_corpus():
    for i in tqdm(range(0, len(dataset), 100)):
        yield dataset[i:i + 100]["full_text"]


raw_tokenizer.train_from_iterator(train_corpus(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

tokenized_texts_test = []
for text in tqdm(test["full_text"].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []
for text in tqdm(train["full_text"].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


def dummy(text):
    return text


vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    lowercase=False,
    sublinear_tf=True,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode",
    min_df=0.05,
    max_df=0.95,
    max_features=1000
)

vectorizer.fit(tokenized_texts_test)
vocab = vectorizer.vocabulary_
print(len(vocab))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    lowercase=False,
    sublinear_tf=True,
    vocabulary=vocab,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode",
    min_df=0.05,
    max_df=0.95,
    max_features=1000
)

X_train = vectorizer.fit_transform(tokenized_texts_train)
X_test = vectorizer.transform(tokenized_texts_test)

print("The shape of X_train is:", X_train.shape)
print("The shape of X_test is:", X_test.shape)

# del vectorizer
gc.collect()


# -------- 生成文本相关特征
essay_processor = EssayProcessor()
train_sent_agg_df = essay_processor.sentence_processor(df=train)
train_paragraph_agg_df = essay_processor.paragraph_processor(df=train)
train_word_agg_df = essay_processor.word_processor(df=train)

test_sent_agg_df = essay_processor.sentence_processor(df=test)
test_paragraph_agg_df = essay_processor.paragraph_processor(df=test)
test_word_agg_df = essay_processor.word_processor(df=test)

train_essay_agg_df = train.merge(train_sent_agg_df, on="essay_id", how="left")
train_essay_agg_df = train_essay_agg_df.merge(
    train_paragraph_agg_df, on="essay_id", how="left"
)
train_essay_agg_df = train_essay_agg_df.merge(
    train_word_agg_df, on="essay_id", how="left"
)

test_essay_agg_df = test.merge(test_sent_agg_df, on="essay_id", how="left")
test_essay_agg_df = test_essay_agg_df.merge(
    test_paragraph_agg_df, on="essay_id", how="left"
)
test_essay_agg_df = test_essay_agg_df.merge(
    test_word_agg_df, on="essay_id", how="left"
)

train_essay_agg_df.drop(["word", "sent", "paragraph"], axis=1, inplace=True)
test_essay_agg_df.drop(["word", "sent", "paragraph"], axis=1, inplace=True)

print("The shape of train_essay_agg_df is:", train_essay_agg_df.shape)
print(train_essay_agg_df.dtypes)
print("The shape of test_essay_agg_df is:", test_essay_agg_df.shape)

X_train = pd.DataFrame(X_train.toarray())
X_train_tfidf_columns = [f"tfid_{i}" for i in range(len(X_train.columns))]
X_train.columns = X_train_tfidf_columns

X_test = pd.DataFrame(X_test.toarray())
X_test_tfidf_columns = [f"tfid_{i}" for i in range(len(X_test.columns))]
X_test.columns = X_test_tfidf_columns

X_train = pd.concat([X_train, train_essay_agg_df], axis=1)
X_train = train_essay_agg_df
X_train.drop(["essay_id", "full_text"], axis=1, inplace=True)
# X_train.dropna(inplace=True)
# X_train = pd.DataFrame({col: X_train[col].astype(float) for col in X_train.columns})
X_train.fillna(0.0, inplace=True)
has_null = X_train.isnull().values.any()
print(has_null)

X_test = pd.concat([X_test, test_essay_agg_df], axis=1)
X_test = test_essay_agg_df
X_test.drop(["essay_id", "full_text"], axis=1, inplace=True)
# X_test.dropna(inplace=True)
# X_test = pd.DataFrame({col: X_test[col].astype(float) for col in X_test.columns})
X_test.fillna(0.0, inplace=True)
has_null = X_test.isnull().values.any()
print(has_null)

print("The shape of final X_train is:", X_train.shape)
print("The shape of final X_test is:", X_test.shape)


# -------- 准备训练
print(X_train.dtypes)

# X_train["labels"] = X_train.score.map(lambda x: x - 1)
# X_train.drop(["score"], axis=1, inplace=True)


def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - a) ** 2 + b)
    df = preds - labels
    dg = preds - a
    grad = (df / g - f * dg / g ** 2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 0.0
b = 0.0


oof = []
models = []
X = X_train
y = X_train["score"].values
feature_names = list(filter(lambda x: x not in ["score"], X_train.columns))

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
callbacks = [
    log_evaluation(period=50),
    early_stopping(stopping_rounds=500, first_metric_only=True)
]
for fold_id, (train_idx, val_idx) in tqdm(enumerate(skf.split(X.copy(), y.copy().astype(str))), total=5):
    model = lgb.LGBMRegressor(
        objective=qwk_obj,
        metrics="None",
        learning_rate=0.005,
        n_estimators=10000,
        max_depth=17,
        num_leaves=15,
        reg_alpha=0.2,
        reg_lambda=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbosity=-1
    )
    X_train_tmp = X_train.iloc[train_idx][feature_names]
    y_train_tmp = X_train.iloc[train_idx]["score"] - a
    X_val_tmp = X_train.iloc[val_idx][feature_names]
    y_val_tmp = X_train.iloc[val_idx]["score"] - a

    print("\n==== Fold_{} Training ====\n".format(fold_id + 1))
    lgb_model = model.fit(
        X_train_tmp,
        y_train_tmp,
        eval_names=["train", "valid"],
        eval_set=[(X_train_tmp, y_train_tmp), (X_val_tmp, y_val_tmp)],
        eval_metric=quadratic_weighted_kappa,
        callbacks=callbacks
    )

    pred_val = lgb_model.predict(
        X_val_tmp,
        num_iteration=lgb_model.best_iteration_
    )
    df_tmp = X_train.iloc[val_idx][["score"]].copy()
    df_tmp["pred"] = pred_val + a

    oof.append(df_tmp)
    models.append(model.booster_)
    lgb_model.booster_.save_model(f"lgbm/fold_{fold_id}.txt")

df_oof = pd.concat(oof)

acc = accuracy_score(df_oof["score"], df_oof["pred"].clip(1, 6).round())
kappa = cohen_kappa_score(
    df_oof["score"],
    df_oof["pred"].clip(1, 6).round(),
    weights="quadratic"
)
print("Acc:", acc)
print("Kappa:", kappa)

reloaded_test = pd.read_csv("../dataset/train.csv")
prediction = reloaded_test[["essay_id"]].copy()
prediction["score"] = 0
pred_test = models[0].predict(X_test[feature_names]) + a
for i in range(4):
    pred_now = models[i + 1].predict(X_test[feature_names]) + a
    pred_test = np.add(pred_test, pred_now)
pred_test = pred_test / 5.0
print(pred_test)

pred_test = pred_test.clip(1, 6).round()
prediction["score"] = pred_test
prediction.to_csv("submission.csv", index=False)
print(prediction.head(3))
