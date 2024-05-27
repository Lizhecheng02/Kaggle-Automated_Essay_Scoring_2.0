import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, cohen_kappa_score
from config import CFG


def quadratic_weighted_kappa(y_true, y_pred):
    if isinstance(y_pred, xgb.QuantileDMatrix):
        y_true, y_pred = y_pred, y_true
        y_true = (y_true.get_label() + CFG.a).round()
        y_pred = (y_pred + CFG.a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return "QWK", qwk
    else:
        y_true = (y_true + CFG.a).round()
        y_pred = (y_pred + CFG.a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + CFG.a
    preds = y_pred + CFG.a
    preds = preds.clip(1, 6)
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - CFG.a) ** 2 + CFG.b)
    df = preds - labels
    dg = preds - CFG.a
    grad = (df / g - f * dg / g ** 2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess
