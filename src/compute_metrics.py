from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(p):
    preds, labels = p
    score = cohen_kappa_score(
        labels,
        preds.clip(1, 6).round(),
        weights="quadratic"
    )
    return {"qwk": score}
