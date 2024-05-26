class CFG:
    random_state = 3047

    # train file path
    train_file_path = "../dataset/train.csv"

    # test file path
    test_file_path = "../dataset/test.csv"

    # use out of fold cv
    main_file_path = "../dataset/13k_overlap.csv"
    out_of_fold_file_path = "../dataset/4k_nooverlap.csv"

    # use feedback features
    train_feedback_file_path = "../dataset/train_feedback_features.csv"
    test_feedback_file_path = "../dataset/test_feedback_features.csv"

    # whether to preprocess text
    DO_PREPROCESS = False
    word_file_path = "words.txt"

    # whether to train new tokenizer for tfidf vectorizer
    TRAIN_TOKENIZER = False

    # whether to use original tfidf vectorizer
    USE_ORIGINAL_TFIDF_VECTORIZER = True

    # whether to use original count vectorizer
    USE_ORIGINAL_COUNT_VECTORIZER = True

    # whether to use essay processor
    USE_EssayProcessor = True

    # whether to use feedback features
    USE_FEEDBACK_FEATURES = True

    # train own tokenizer
    LOWER_CASE = False
    VOCAB_SIZE = 32000
    tokenizer_training_ngram_range = (1, 3)
    tokenizer_training_min_df = 0.05
    tokenizer_training_max_df = 0.95
    tokenizer_training_max_features = 2000

    # a and b for metrics
    # 17k
    a = 2.998 
    b = 1.092
    # 19k
    # a = 2.99575151546552
    # b = 0.9725680876911655

    # original tf-idf vectorizer
    tfidf_vectorizer_ngram_range = (2, 3)
    tfidf_vectorizer_min_df = 0.10
    tfidf_vectorizer_max_df = 0.90
    tfidf_vectorizer_max_features = 2000

    # count vectorizer
    count_vectorizer_ngram_range = (2, 3)
    count_vectorizer_min_df = 0.10
    count_vectorizer_max_df = 0.90
    count_vectorizer_max_features = 2000

    # parameters for lgbm
    lgbm_n_split = 5
    lgbm_log_evaluation = 25
    lgbm_stopping_rounds = 100
    lgbm_learning_rate = 0.01
    lgbm_n_estimators = 2048
    lgbm_max_depth = 10
    lgbm_num_leaves = 15
    lgbm_reg_alpha = 2
    lgbm_reg_lambda = 0.1
    lgbm_colsample_bytree = 0.3
    lgbm_verbosity = 1

    # parameters for xgboost
    xgb_n_split = 5
    xgb_log_evaluation = 25
    xgb_stopping_rounds = 100
    xgb_learning_rate = 0.01
    xgb_n_estimators = 2048
    xgb_max_depth = 10
    xgb_num_leaves = 15
    xgb_reg_alpha = 2
    xgb_reg_lambda = 0.1
    xgb_colsample_bytree = 0.3
    xgb_verbosity = 1
