from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from transformers import PreTrainedTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
from tqdm import tqdm
from config import CFG
import pandas as pd


def train_tokenizer(train, test):

    # -------- 训练tokenizer ---------
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
        ngram_range=CFG.tokenizer_training_ngram_range,
        lowercase=False,
        sublinear_tf=True,
        analyzer="word",
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents="unicode",
        min_df=CFG.tokenizer_training_min_df,
        max_df=CFG.tokenizer_training_max_df,
        max_features=CFG.tokenizer_training_max_features
    )

    vectorizer.fit(tokenized_texts_test)
    vocab = vectorizer.vocabulary_
    print("The length of vocabulary for trained new tokenizer is:", len(vocab))

    vectorizer = TfidfVectorizer(
        ngram_range=CFG.tokenizer_training_ngram_range,
        lowercase=False,
        sublinear_tf=True,
        vocabulary=vocab,
        analyzer="word",
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents="unicode",
        min_df=CFG.tokenizer_training_min_df,
        max_df=CFG.tokenizer_training_max_df,
        max_features=CFG.tokenizer_training_max_features
    )

    X_train = vectorizer.fit_transform(tokenized_texts_train)
    dense_matrix = X_train.toarray()
    df_train = pd.DataFrame(dense_matrix)
    tfidf_columns = [f"new_tokenizer_tfidf_{i}" for i in range(len(df_train.columns))]
    df_train.columns = tfidf_columns
    df_train["essay_id"] = train["essay_id"]

    del dense_matrix, tfidf_columns

    X_test = vectorizer.transform(tokenized_texts_test)
    dense_matrix = X_test.toarray()
    df_test = pd.DataFrame(dense_matrix)
    tfidf_columns = [f"new_tokenizer_tfidf_{i}" for i in range(len(df_test.columns))]
    df_test.columns = tfidf_columns
    df_test["essay_id"] = train["essay_id"]

    del dense_matrix, tfidf_columns
    del X_train, X_test

    print("The shape of X_train after training new tokenizer and vectorizer is:", df_train.shape)
    print("The shape of X_test after training new tokenizer and vectorizer is:", df_test.shape)

    return tokenizer, df_train, df_test
