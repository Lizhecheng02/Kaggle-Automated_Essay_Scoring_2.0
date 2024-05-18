class CFG:
    # whether use autoclassification
    use_autoclassification = False

    # you can choose from "mean_pooling", "meanmax_pooling", "attention_pooling", "weighted_layer_pooling", "concat_pooling", "lstm_pooling", "gru_pooling"
    pooling_type = "mean_pooling"
    backbone_model = "microsoft/deberta-v3-small"
    max_length = 1536
    use_wandb = True

    # regression
    zero_dropout = True

    # whether to use rubric
    use_rubric = True

    # replace \n\n
    is_replace = True

    # cross validation, you can choose from "only4k", "only13k", "only17k", "only19k", "only30k", "train17k_validation4k"
    validation_type = "train17k_validation4k"
    num_split = 4
    selected_fold_id = 0

    # attention_pooling
    hiddendim_fc = 256
    dropout = 0.0

    # weighted_layer_pooling
    layer_start = 6

    # concat_pooling
    num_pooling_layers = 6

    # lstm_pooling or gru_pooling
    hidden_lstm_size = 64
    dropout_rate = 0.1
    bidirectional = True

    # awp
    # train_type = "awp"
    awp_lr = 0.1
    awp_eps = 1.0e-4
    awp_start_epoch = 1.0

    # no awp
    train_type = "no-awp"

    # training_arguments
    learning_rate = 2.0e-5
    per_device_train_batch_size = 2
    per_device_eval_batch_size = per_device_train_batch_size * 2
    gradient_accumulation_steps = 16 // per_device_train_batch_size
    num_train_epochs = 3
    weight_decay = 0.001
    steps = 100
    save_total_limit = 5

    # scheduler, you can choose from "linear", "cosine", "polynomial", "constant"
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # if you use polynomial scheduler
    power = 2.0
    lr_end = 3.0e-6
