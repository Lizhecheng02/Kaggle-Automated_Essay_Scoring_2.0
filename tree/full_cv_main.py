import gc
import pandas as pd
import warnings
from train_tokenizer import train_tokenizer
from config import CFG
from preprocess import essay_preprocess
from vectorizers import *
from essay_processor import EssayProcessor
from lgbm_train import train_lgbm
warnings.filterwarnings("ignore")

train = pd.read_csv(CFG.train_file_path)
train = train[:1000]
test = pd.read_csv(CFG.test_file_path)

if CFG.DO_PREPROCESS:
    print("DO_PREPROCESS!!!")
    train = essay_preprocess(df=train)
    test = essay_preprocess(df=test)
    train.drop(columns=["full_text_no_punc"], axis=1, inplace=True)
    test.drop(columns=["full_text_no_punc"], axis=1, inplace=True)

if CFG.USE_EssayProcessor:
    print("USE_EssayProcessor!!!")
    essay_processor = EssayProcessor()
    train_sent_agg_df = essay_processor.sentence_processor(df=train)
    train_paragraph_agg_df = essay_processor.paragraph_processor(df=train)
    train_word_agg_df = essay_processor.word_processor(df=train)

    test_sent_agg_df = essay_processor.sentence_processor(df=test)
    test_paragraph_agg_df = essay_processor.paragraph_processor(df=test)
    test_word_agg_df = essay_processor.word_processor(df=test)

    train_essay_agg_df = train.merge(train_sent_agg_df, on="essay_id", how="left")
    train_essay_agg_df = train_essay_agg_df.merge(train_paragraph_agg_df, on="essay_id", how="left")
    train_essay_agg_df = train_essay_agg_df.merge(train_word_agg_df, on="essay_id", how="left")

    test_essay_agg_df = test.merge(test_sent_agg_df, on="essay_id", how="left")
    test_essay_agg_df = test_essay_agg_df.merge(test_paragraph_agg_df, on="essay_id", how="left")
    test_essay_agg_df = test_essay_agg_df.merge(test_word_agg_df, on="essay_id", how="left")

    train_essay_agg_df.drop(columns=["word", "sent", "paragraph", "essay_id", "full_text", "score"], axis=1, inplace=True)
    test_essay_agg_df.drop(columns=["word", "sent", "paragraph", "essay_id", "full_text"], axis=1, inplace=True)

    print("The shape of train_essay_agg_df is:", train_essay_agg_df.shape)
    print("The shape of test_essay_agg_df is:", test_essay_agg_df.shape)

    train.drop(columns=["word", "sent", "paragraph"], axis=1, inplace=True)
    train = pd.concat([train, train_essay_agg_df], axis=1)
    print("The shape of train after essay processor is:", train.shape)
    test.drop(columns=["word", "sent", "paragraph"], axis=1, inplace=True)
    test = pd.concat([test, test_essay_agg_df], axis=1)
    print("The shape of test after essay processor is:", test.shape)

if CFG.TRAIN_TOKENIZER:
    print("TRAIN_TOKENIZER!!!")
    tokenizer, df_train, df_test = train_tokenizer(train=train, test=test)
    train = pd.merge(train, df_train, on="essay_id", how="left")
    print("The shape of train after training new tokenizer is:", train.shape)
    test = pd.merge(test, df_test, on="essay_id", how="left")
    print("The shape of test after training new tokenizer is:", test.shape)

if CFG.USE_ORIGINAL_TFIDF_VECTORIZER:
    print("USE_ORIGINAL_TFIDF_VECTORIZER!!!")
    tf_idf_vectorizer, df_train = use_tf_idf_vectorizer(train=train)
    train = pd.merge(train, df_train, on="essay_id", how="left")
    print("The shape of train after using original tf-idf vectorizer is:", train.shape)
    df_test = use_tf_idf_vectorizer_for_test(vectorizer=tf_idf_vectorizer, test=test)
    test = pd.merge(test, df_test, on="essay_id", how="left")
    print("The shape of test after using original tf-idf vectorizer is:", test.shape)

if CFG.USE_ORIGINAL_COUNT_VECTORIZER:
    print("USE_ORIGINAL_COUNT_VECTORIZER!!!")
    count_vectorizer, df_train = use_count_vectorizer(train=train)
    train = pd.merge(train, df_train, on="essay_id", how="left")
    print("The shape of train after using original count vectorizer is:", train.shape)
    df_test = use_count_vectorizer_for_test(vectorizer=count_vectorizer, test=test)
    test = pd.merge(test, df_test, on="essay_id", how="left")
    print("The shape of test after using original count vectorizer is:", test.shape)

train.drop(["essay_id", "full_text"], axis=1, inplace=True)
train.fillna(0.0, inplace=True)
has_null_train = train.isnull().values.any()
print(has_null_train)

test.drop(["essay_id", "full_text"], axis=1, inplace=True)
test.fillna(0.0, inplace=True)
has_null_test = test.isnull().values.any()
print(has_null_test)

print("The shape of final train is:", train.shape)
print("The shape of final test is:", test.shape)

gc.collect()

lgbm_models, lgbm_model_nums, lgbm_predictions = train_lgbm(X_train=train, X_test=test)
