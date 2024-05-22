from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from config import CFG
import pandas as pd
import pickle


def tokenizer(x):
    return x


def preprocessor(x):
    return x


def use_tf_idf_vectorizer(train):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        token_pattern=None,
        strip_accents="unicode",
        analyzer="word",
        sublinear_tf=True,
        ngram_range=CFG.tfidf_vectorizer_ngram_range,
        min_df=CFG.tfidf_vectorizer_min_df,
        max_df=CFG.tfidf_vectorizer_max_df,
        max_features=CFG.tfidf_vectorizer_max_features
    )

    X_train = vectorizer.fit_transform([i for i in train["full_text"]])
    dense_matrix = X_train.toarray()
    df = pd.DataFrame(dense_matrix)
    tfidf_columns = [f"tfidf_{i}" for i in range(len(df.columns))]
    df.columns = tfidf_columns
    df["essay_id"] = train["essay_id"]

    with open("tf_idf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer, df


def use_count_vectorizer(train):
    vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        token_pattern=None,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=CFG.count_vectorizer_ngram_range,
        min_df=CFG.count_vectorizer_min_df,
        max_df=CFG.count_vectorizer_max_df,
    )

    X_train = vectorizer.fit_transform([i for i in train["full_text"]])
    dense_matrix = X_train.toarray()
    df = pd.DataFrame(dense_matrix)
    count_columns = [f"count_{i}" for i in range(len(df.columns))]
    df.columns = count_columns
    df["essay_id"] = train["essay_id"]

    with open("count_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer, df


def use_tf_idf_vectorizer_for_test(vectorizer, test):
    X_test = vectorizer.transform([i for i in test["full_text"]])
    dense_matrix = X_test.toarray()
    df = pd.DataFrame(dense_matrix)
    tfidf_columns = [f"tfidf_{i}" for i in range(len(df.columns))]
    df.columns = tfidf_columns
    df["essay_id"] = test["essay_id"]
    return df


def use_count_vectorizer_for_test(vectorizer, test):
    X_test = vectorizer.transform([i for i in test["full_text"]])
    dense_matrix = X_test.toarray()
    df = pd.DataFrame(dense_matrix)
    count_columns = [f"count_{i}" for i in range(len(df.columns))]
    df.columns = count_columns
    df["essay_id"] = test["essay_id"]
    return df
