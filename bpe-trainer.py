import pandas as pd
import gc
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from tqdm import tqdm
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer


class CFG:
    random_state = 42
    LOWER_CASE = False
    VOCAB_SIZE = 30522


train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

# 训练tokenizer
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
    strip_accents="unicode"
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
    strip_accents="unicode"
)

X_train = vectorizer.fit_transform(tokenized_texts_train)
X_test = vectorizer.transform(tokenized_texts_test)

print("The shape of X_train is:", X_train.shape)
print("The shape of X_test is:", X_test.shape)

del vectorizer
gc.collect()
