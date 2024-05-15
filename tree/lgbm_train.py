import gc
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from metrics import qwk_obj, quadratic_weighted_kappa
from lightgbm import log_evaluation, early_stopping
from tqdm import tqdm
from config import CFG
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")


def train_lgbm(X_train, X_test):
    oof = []
    models = []

    X = X_train
    y = X_train["score"].values
    feature_names = list(filter(lambda x: x not in ["score"], X_train.columns))

    reloaded_test = pd.read_csv(CFG.test_file_path)
    prediction = reloaded_test[["essay_id"]].copy()

    skf = StratifiedKFold(
        n_splits=CFG.lgbm_n_split,
        random_state=CFG.random_state,
        shuffle=True
    )

    callbacks = [
        log_evaluation(period=CFG.lgbm_log_evaluation),
        early_stopping(
            stopping_rounds=CFG.lgbm_stopping_rounds,
            first_metric_only=True
        )
    ]
    for fold_id, (train_idx, val_idx) in tqdm(enumerate(skf.split(X.copy(), y.copy().astype(str))), total=5):
        model = lgb.LGBMRegressor(
            objective=qwk_obj,
            metrics="None",
            learning_rate=CFG.lgbm_learning_rate,
            n_estimators=CFG.lgbm_n_estimators,
            max_depth=CFG.lgbm_max_depth,
            num_leaves=CFG.lgbm_num_leaves,
            reg_alpha=CFG.lgbm_reg_alpha,
            reg_lambda=CFG.lgbm_reg_lambda,
            colsample_bytree=CFG.lgbm_colsample_bytree,
            random_state=CFG.random_state,
            verbosity=CFG.lgbm_verbosity
        )
        X_train_tmp = X_train.iloc[train_idx][feature_names]
        y_train_tmp = X_train.iloc[train_idx]["score"] - CFG.a
        X_val_tmp = X_train.iloc[val_idx][feature_names]
        y_val_tmp = X_train.iloc[val_idx]["score"] - CFG.a

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
        df_tmp["pred"] = pred_val + CFG.a

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

    prediction["score"] = 0
    pred_test = models[0].predict(X_test[feature_names]) + CFG.a
    for i in range(CFG.lgbm_n_split - 1):
        pred_now = models[i + 1].predict(X_test[feature_names]) + CFG.a
        pred_test = np.add(pred_test, pred_now)
    pred_test = pred_test / 5.0
    pred_test = pred_test.clip(1, 6).round()
    print(pred_test)

    prediction["score"] = pred_test
    prediction.to_csv("submission_lgbm.csv", index=False)
    print(prediction.head(3))

    return models, len(models), prediction


def train_lgbm_out_of_fold(X_train_main, X_train_out_of_fold, X_test):
    oof = []
    models = []

    feature_names = list(filter(lambda x: x not in ["score"], X_train_main.columns))

    X_out_of_fold = X_train_out_of_fold
    y_out_of_fold = X_train_out_of_fold["score"].values

    reloaded_test = pd.read_csv(CFG.test_file_path)
    prediction = reloaded_test[["essay_id"]].copy()

    skf = StratifiedKFold(
        n_splits=CFG.lgbm_n_split,
        random_state=CFG.random_state,
        shuffle=True
    )

    callbacks = [
        log_evaluation(period=CFG.lgbm_log_evaluation),
        early_stopping(
            stopping_rounds=CFG.lgbm_stopping_rounds,
            first_metric_only=True
        )
    ]
    for fold_id, (train_idx, val_idx) in tqdm(enumerate(skf.split(X_out_of_fold.copy(), y_out_of_fold.copy().astype(str))), total=5):
        model = lgb.LGBMRegressor(
            objective=qwk_obj,
            metrics="None",
            learning_rate=CFG.lgbm_learning_rate,
            n_estimators=CFG.lgbm_n_estimators,
            max_depth=CFG.lgbm_max_depth,
            num_leaves=CFG.lgbm_num_leaves,
            reg_alpha=CFG.lgbm_reg_alpha,
            reg_lambda=CFG.lgbm_reg_lambda,
            colsample_bytree=CFG.lgbm_colsample_bytree,
            random_state=CFG.random_state,
            verbosity=CFG.lgbm_verbosity
        )
        X_train_tmp = X_train_out_of_fold.iloc[train_idx][feature_names]
        y_train_tmp = X_train_out_of_fold.iloc[train_idx]["score"] - CFG.a

        X_main = X_train_main[feature_names]
        X_train_tmp = pd.concat([X_train_tmp, X_main], axis=0)
        y_main = X_train_main["score"] - CFG.a
        y_train_tmp = pd.concat([y_train_tmp, y_main], axis=0)

        X_val_tmp = X_train_out_of_fold.iloc[val_idx][feature_names]
        y_val_tmp = X_train_out_of_fold.iloc[val_idx]["score"] - CFG.a

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
        df_tmp = X_out_of_fold.iloc[val_idx][["score"]].copy()
        df_tmp["pred"] = pred_val + CFG.a

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

    prediction["score"] = 0
    pred_test = models[0].predict(X_test[feature_names]) + CFG.a
    for i in range(CFG.lgbm_n_split - 1):
        pred_now = models[i + 1].predict(X_test[feature_names]) + CFG.a
        pred_test = np.add(pred_test, pred_now)
    pred_test = pred_test / 5.0
    pred_test = pred_test.clip(1, 6).round()
    print(pred_test)

    prediction["score"] = pred_test
    prediction.to_csv("submission_lgbm.csv", index=False)
    print(prediction.head(3))

    return models, len(models), prediction
