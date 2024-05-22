import gc
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from config import CFG
from preprocess import essay_preprocess
from vectorizers import *
from essay_processor import EssayProcessor
warnings.filterwarnings("ignore")

test = pd.read_csv(CFG.test_file_path)
test_essay_ids = test["essay_id"]

if CFG.DO_PREPROCESS:
    print("DO_PREPROCESS!!!")
    test = essay_preprocess(df=test)
    test.drop(columns=["full_text_no_punc"], axis=1, inplace=True)

if CFG.USE_EssayProcessor:
    print("USE_EssayProcessor!!!")
    essay_processor = EssayProcessor()

    test_sent_agg_df = essay_processor.sentence_processor(df=test)
    test_paragraph_agg_df = essay_processor.paragraph_processor(df=test)
    test_word_agg_df = essay_processor.word_processor(df=test)

    test_essay_agg_df = test.merge(test_sent_agg_df, on="essay_id", how="left")
    test_essay_agg_df = test_essay_agg_df.merge(test_paragraph_agg_df, on="essay_id", how="left")
    test_essay_agg_df = test_essay_agg_df.merge(test_word_agg_df, on="essay_id", how="left")

    test_essay_agg_df.drop(columns=["word", "sent", "paragraph", "essay_id", "full_text"], axis=1, inplace=True)

    print("The shape of test_essay_agg_df is:", test_essay_agg_df.shape)

    test.drop(columns=["word", "sent", "paragraph"], axis=1, inplace=True)
    test = pd.concat([test, test_essay_agg_df], axis=1)
    print("The shape of test after essay processor is:", test.shape)

    test["essay_id"] = test_essay_ids

# 如果要离线推理，就不能训练新的tokenizer

if CFG.USE_ORIGINAL_TFIDF_VECTORIZER:
    print("USE_ORIGINAL_TFIDF_VECTORIZER!!!")

    with open("tf_idf_vectorizer.pkl", "rb") as f:
        tf_idf_vectorizer = pickle.load(f)

    df_test = use_tf_idf_vectorizer_for_test(vectorizer=tf_idf_vectorizer, test=test)
    test = pd.merge(test, df_test, on="essay_id", how="left")
    print("The shape of test after using original tf-idf vectorizer is:", test.shape)

if CFG.USE_ORIGINAL_COUNT_VECTORIZER:
    print("USE_ORIGINAL_COUNT_VECTORIZER!!!")

    with open("count_vectorizer.pkl", "rb") as f:
        count_vectorizer = pickle.load(f)

    df_test = use_count_vectorizer_for_test(vectorizer=count_vectorizer, test=test)
    test = pd.merge(test, df_test, on="essay_id", how="left")
    print("The shape of test after using original count vectorizer is:", test.shape)

if CFG.USE_FEEDBACK_FEATURES:
    print("USE_FEEDBACK_FEATURES!!!")
    feedback_df_test = pd.read_csv(CFG.test_feedback_file_path)
    test = pd.merge(test, feedback_df_test, on="essay_id", how="left")
    print("The shape of test after using feedback features is:", test.shape)


test.drop(["essay_id", "full_text"], axis=1, inplace=True)
test.fillna(0.0, inplace=True)
has_null_test = test.isnull().values.any()
print(has_null_test)

print("The shape of final test is:", test.shape)

gc.collect()

all_preds = []
for fold_id in range(5):
    print(f"=========== fold_{fold_id} ============")
    model = lgb.Booster(model_file=f"lgbm/fold_{fold_id}.txt")
    y_pred = model.predict(test)
    all_preds.append(y_pred)
    print(y_pred)

all_preds_concat = np.concatenate(all_preds, axis=0)
avg_preds = np.mean(all_preds_concat, axis=0)
clipped_preds = np.clip(avg_preds, 1, 6)
final_list = clipped_preds.tolist()
print(final_list)

test = pd.read_csv(CFG.test_file_path)
test["score"] = final_list
test = test[["essay_id", "score"]]
print(test.head())
