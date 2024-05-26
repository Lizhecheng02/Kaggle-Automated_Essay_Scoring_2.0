import gc
import torch
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from metrics import qwk_obj, quadratic_weighted_kappa
from tqdm import tqdm
from config import CFG
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")


def train_xgb(X_train, X_test):
    oof = []
    models = []

    X = X_train
    y = X_train["score"].values
    feature_names = list(filter(lambda x: x not in ["score"], X_train.columns))

    reloaded_test = pd.read_csv(CFG.test_file_path)
    prediction = reloaded_test[["essay_id"]].copy()

    skf = StratifiedKFold(
        n_splits=CFG.xgb_n_split,
        random_state=CFG.random_state,
        shuffle=True
    )

    callbacks = [
        # xgb.callback.EvaluationMonitor(period=CFG.xgb_log_evaluation),
        xgb.callback.EarlyStopping(
            CFG.xgb_stopping_rounds, metric_name="QWK",
            maximize=True, save_best=True
        )
    ]

    for fold_id, (train_idx, val_idx) in tqdm(enumerate(skf.split(X.copy(), y.copy().astype(str))), total=5):
        model = xgb.XGBRegressor(
            objective=qwk_obj,
            metrics="None",
            learning_rate=CFG.xgb_learning_rate,
            n_estimators=CFG.xgb_n_estimators,
            max_depth=CFG.xgb_max_depth,
            num_leaves=CFG.xgb_num_leaves,
            reg_alpha=CFG.xgb_reg_alpha,
            reg_lambda=CFG.xgb_reg_lambda,
            colsample_bytree=CFG.xgb_colsample_bytree,
            random_state=CFG.random_state,
            verbosity=CFG.xgb_verbosity,
            # extra_trees=True,
            # class_weight="balanced",
            # tree_method="hist",
            device="gpu" if torch.cuda.is_available() else "cpu"
        )
        X_train_tmp = X_train.iloc[train_idx][feature_names]
        y_train_tmp = X_train.iloc[train_idx]["score"] - CFG.a
        X_val_tmp = X_train.iloc[val_idx][feature_names]
        y_val_tmp = X_train.iloc[val_idx]["score"] - CFG.a

        print("\n==== Fold_{} Training ====\n".format(fold_id + 1))

        try:
            xgb_model = model.fit(
                X_train_tmp,
                y_train_tmp,
                eval_set=[(X_train_tmp, y_train_tmp), (X_val_tmp, y_val_tmp)],
                eval_metric=quadratic_weighted_kappa,
                callbacks=callbacks
            )
            best_iteration = xgb_model.best_iteration
            print(f"Best iteration for fold {fold_id + 1}: {best_iteration}")
        except:
            best_iteration = None
            print(f"Early stopping not triggered for fold {fold_id + 1}")

        pred_val = xgb_model.predict(X_val_tmp)
        df_tmp = X_train.iloc[val_idx][["score"]].copy()
        df_tmp["pred"] = pred_val + CFG.a

        oof.append(df_tmp)
        models.append(model)
        xgb_model.save_model(f"xgb/fold_{fold_id}.model")

        del model, xgb_model

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
    for i in range(CFG.xgb_n_split - 1):
        pred_now = models[i + 1].predict(X_test[feature_names]) + CFG.a
        pred_test = np.add(pred_test, pred_now)
    pred_test = pred_test / 5.0
    pred_test = pred_test.clip(1, 6).round()
    print(pred_test)

    prediction["score"] = pred_test
    prediction.to_csv("submission_xgb.csv", index=False)
    print(prediction.head(3))

    return models, len(models), prediction


def train_xgb_out_of_fold(X_train_main, X_train_out_of_fold, X_test):
    oof = []
    models = []

    feature_names = list(filter(lambda x: x not in ["score"], X_train_main.columns))

    X_out_of_fold = X_train_out_of_fold
    y_out_of_fold = X_train_out_of_fold["score"].values

    reloaded_test = pd.read_csv(CFG.test_file_path)
    prediction = reloaded_test[["essay_id"]].copy()

    skf = StratifiedKFold(
        n_splits=CFG.xgb_n_split,
        random_state=CFG.random_state,
        shuffle=True
    )

    callbacks = [
        # xgb.callback.EvaluationMonitor(period=CFG.xgb_log_evaluation),
        xgb.callback.EarlyStopping(
            CFG.xgb_stopping_rounds, metric_name="QWK",
            maximize=True, save_best=True
        )
    ]
    for fold_id, (train_idx, val_idx) in tqdm(enumerate(skf.split(X_out_of_fold.copy(), y_out_of_fold.copy().astype(str))), total=5):
        model = xgb.xgbRegressor(
            objective=qwk_obj,
            metrics="None",
            learning_rate=CFG.xgb_learning_rate,
            n_estimators=CFG.xgb_n_estimators,
            max_depth=CFG.xgb_max_depth,
            num_leaves=CFG.xgb_num_leaves,
            reg_alpha=CFG.xgb_reg_alpha,
            reg_lambda=CFG.xgb_reg_lambda,
            colsample_bytree=CFG.xgb_colsample_bytree,
            random_state=CFG.random_state,
            verbosity=CFG.xgb_verbosity,
            extra_trees=True,
            class_weight="balanced",
            tree_method="hist",
            device="gpu" if torch.cuda.is_available() else "cpu"
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
        xgb_model = model.fit(
            X_train_tmp,
            y_train_tmp,
            eval_set=[(X_train_tmp, y_train_tmp), (X_val_tmp, y_val_tmp)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks
        )

        pred_val = xgb_model.predict(X_val_tmp)
        df_tmp = X_out_of_fold.iloc[val_idx][["score"]].copy()
        df_tmp["pred"] = pred_val + CFG.a

        oof.append(df_tmp)
        models.append(model)
        xgb_model.save_model(f"xgb/fold_{fold_id}.model")

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
    for i in range(CFG.xgb_n_split - 1):
        pred_now = models[i + 1].predict(X_test[feature_names]) + CFG.a
        pred_test = np.add(pred_test, pred_now)
    pred_test = pred_test / 5.0
    pred_test = pred_test.clip(1, 6).round()
    print(pred_test)

    prediction["score"] = pred_test
    prediction.to_csv("submission_xgb.csv", index=False)
    print(prediction.head(3))

    return models, len(models), prediction
