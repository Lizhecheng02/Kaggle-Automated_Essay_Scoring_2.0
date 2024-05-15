class CFG:
    random_state = 3047

    # whether to preprocess text
    DO_PREPROCESS = False
    word_file_path = "words.txt"

    # whether to train new tokenizer for tfidf vectorizer
    TRAIN_TOKENIZER = True

    # whether to use original tfidf vectorizer
    USE_ORIGINAL_TFIDF_VECTORIZER = True

    # whether to use original count vectorizer
    USE_ORIGINAL_COUNT_VECTORIZER = True

    # whether to use essay processor
    USE_EssayProcessor = True

    # train own tokenizer
    LOWER_CASE = False
    VOCAB_SIZE = 32000
    tokenizer_training_ngram_range = (1, 3)
    tokenizer_training_min_df = 0.05
    tokenizer_training_max_df = 0.95
    tokenizer_training_max_features = 100

    # a and b for metrics
    a = 2.998
    b = 1.092

    # original tf-idf vectorizer
    tfidf_vectorizer_ngram_range = (1, 3)
    tfidf_vectorizer_min_df = 0.05
    tfidf_vectorizer_max_df = 0.95
    tfidf_vectorizer_max_features = 100

    # count vectorizer
    count_vectorizer_ngram_range = (1, 3)
    count_vectorizer_min_df = 0.05
    count_vectorizer_max_df = 0.95
    count_vectorizer_max_features = 100

    # parameters for lgbm
    lgbm_n_split = 5
    lgbm_log_evaluation = 50
    lgbm_stopping_rounds = 500
    lgbm_learning_rate = 0.005
    lgbm_n_estimators = 10000
    lgbm_max_depth = 17
    lgbm_num_leaves = 15
    lgbm_reg_alpha = 0.4
    lgbm_reg_lambda = 0.8
    lgbm_colsample_bytree = 0.7
    lgbm_verbosity = 1
