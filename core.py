"""Core CV, model, fidelity, and corrected interval implementation."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

SEED = 42
NUMERIC = ["age", "avg_glucose_level", "bmi"]
BINARY = ["hypertension", "heart_disease", "bmi_was_missing"]
CATEGORICAL = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
BASE_INTERPRETABLE = ["L1 logistic regression", "Depth-limited CART"]
BASE_BLACK_BOX = ["Random forest", "Gradient boosting", "Multilayer perceptron"]


def prepare_data(df):
    """Validate the Kaggle schema and return raw feature frame and binary target."""
    data = df.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    data = data.drop(columns=["id"], errors="ignore")
    required = set(NUMERIC + ["hypertension", "heart_disease"] + CATEGORICAL + ["stroke"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    ordered = NUMERIC + ["hypertension", "heart_disease"] + CATEGORICAL + ["stroke"]
    data = data[ordered].copy()
    target = pd.to_numeric(data.pop("stroke"), errors="raise")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("stroke must contain only non-missing 0 and 1 values")
    data["bmi_was_missing"] = data["bmi"].isna().astype("int8")
    return data, target.astype(int).to_numpy()


def build_preprocessor():
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn <1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), NUMERIC),
        ("binary", "passthrough", BINARY),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", encoder)]), CATEGORICAL),
    ], remainder="drop")


def available_models():
    names = BASE_INTERPRETABLE + BASE_BLACK_BOX
    try:
        from interpret.glassbox import ExplainableBoostingClassifier  # noqa: F401
        names.insert(1, "Explainable Boosting Machine")
    except Exception as exc:
        warnings.warn(f"EBM unavailable; omitting it: {exc}")
    return names


def build_models(seed=SEED):
    models = {
        "L1 logistic regression": LogisticRegression(penalty="l1", solver="liblinear",
            class_weight="balanced", max_iter=3000, random_state=seed),
        "Depth-limited CART": DecisionTreeClassifier(max_depth=4,
            class_weight="balanced", random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=300,
            class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=100,
            learning_rate=0.05, max_depth=2, random_state=seed),
        "Multilayer perceptron": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001,
            max_iter=500, early_stopping=True, random_state=seed),
    }
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        models["Explainable Boosting Machine"] = ExplainableBoostingClassifier(
            interactions=2, outer_bags=8, max_rounds=5000, n_jobs=1, random_state=seed)
    except Exception:
        pass
    return models


def _probability(model, X):
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(X)[:, classes.index(1)]
    scores = model.decision_function(X)
    return scores if np.ndim(scores) == 1 else scores[:, list(model.classes_).index(1)]


def _fit_predict(name, estimator, X_train, y_train, X_test, seed):
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = clone(estimator)
    if name in ("Explainable Boosting Machine", "Gradient boosting", "Multilayer perceptron"):
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_test), dtype=int).ravel()
    return model, pred, _probability(model, X_test)


def run_experiment(X, y, n_splits=10, n_repeats=5, seed=SEED):
    """Run fold-local preprocessing, performance evaluation, and surrogate fidelity."""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Repeated stratified ROC-AUC evaluation requires both target classes")
    models_available = build_models(seed)
    interpretable = [n for n in models_available if n in BASE_INTERPRETABLE or
                     n == "Explainable Boosting Machine"]
    black_box = [n for n in BASE_BLACK_BOX if n in models_available]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    model_rows, fidelity_rows = [], []
    for split_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        xtr_raw, xte_raw = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        prep = build_preprocessor()
        xtr, xte = prep.fit_transform(xtr_raw, ytr), prep.transform(xte_raw)
        fold_models = build_models(seed + split_id)
        bb_predictions = {}
        for name, estimator in fold_models.items():
            try:
                model, pred, score = _fit_predict(name, estimator, xtr, ytr, xte, seed + split_id)
                model_rows.append({"split": split_id, "model": name,
                    "roc_auc": roc_auc_score(yte, score),
                    "average_precision": average_precision_score(yte, score),
                    "sensitivity": recall_score(yte, pred, zero_division=0)})
                if name in black_box:
                    bb_predictions[name] = (np.asarray(model.predict(xtr), dtype=int).ravel(), pred)
            except Exception as exc:
                if name == "Explainable Boosting Machine":
                    warnings.warn(f"{name} failed on split {split_id} and is omitted for this split: {exc}")
                    continue
                raise
        # Only fit surrogates corresponding to interpretable estimators available in this runtime.
        for bb, (target_train, target_test) in bb_predictions.items():
            for surrogate_name in interpretable:
                try:
                    surrogate, surrogate_pred, _ = _fit_predict(surrogate_name,
                        build_models(seed + split_id)[surrogate_name],
                        xtr, target_train, xte, seed + split_id + 1000)
                    wrong = target_test != yte
                    mode = int(pd.Series(target_train).mode().iloc[0])
                    baseline = np.full(len(yte), mode)
                    fidelity_rows.append({"split": split_id, "black_box": bb,
                        "surrogate": surrogate_name,
                        "agreement": accuracy_score(target_test, surrogate_pred),
                        "error_agreement": (float(np.mean(surrogate_pred[wrong] == target_test[wrong]))
                                            if wrong.any() else np.nan),
                        "majority_baseline_agreement": accuracy_score(target_test, baseline),
                        "black_box_error_count": int(wrong.sum())})
                except Exception as exc:
                    if surrogate_name == "Explainable Boosting Machine":
                        warnings.warn(f"{surrogate_name} surrogate failed on split {split_id}: {exc}")
                        continue
                    raise
        print(f"Completed split {split_id}/{n_splits * n_repeats}")
    return pd.DataFrame(model_rows), pd.DataFrame(fidelity_rows)


def corrected_resampled_ci(values, n_splits=10, n_repeats=5, test_train_ratio=None,
                           confidence=0.95):
    """Nadeau–Bengio corrected resampled t interval for a vector of fold values."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(vals.mean())
    if len(vals) < 2:
        return {"mean": mean, "ci_low": np.nan, "ci_high": np.nan, "n": len(vals)}
    ratio = test_train_ratio if test_train_ratio is not None else 1 / (n_splits - 1)
    correction = 1 / len(vals) + ratio
    se = np.sqrt(correction * vals.var(ddof=1))
    critical = t.ppf((1 + confidence) / 2, df=len(vals) - 1)
    return {"mean": mean, "ci_low": float(mean - critical * se),
            "ci_high": float(mean + critical * se), "n": len(vals)}


def summarize_results(model_rows, fidelity_rows, n_splits=10, n_repeats=5):
    performance = model_rows.groupby("model").agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        sensitivity_mean=("sensitivity", "mean"), n_folds=("split", "count"))
    comparisons = []
    if not model_rows.empty:
        wide = model_rows.pivot(index="split", columns="model", values="roc_auc")
        interpretable = [n for n in ["L1 logistic regression", "Explainable Boosting Machine",
            "Depth-limited CART"] if n in wide]
        black_box = [n for n in BASE_BLACK_BOX if n in wide]
        for a in interpretable:
            for b in black_box:
                ci = corrected_resampled_ci(wide[a] - wide[b], n_splits, n_repeats)
                comparisons.append({"interpretable_model": a, "black_box_model": b,
                                    "mean_difference": ci["mean"], "ci_low": ci["ci_low"],
                                    "ci_high": ci["ci_high"], "n_folds": ci["n"]})
    auc_comparisons = pd.DataFrame(comparisons)
    fidelity_summaries = []
    if not fidelity_rows.empty:
        for (bb, surrogate), group in fidelity_rows.groupby(["black_box", "surrogate"]):
            row = {"black_box": bb, "surrogate": surrogate}
            for metric in ("agreement", "error_agreement"):
                ci = corrected_resampled_ci(group[metric], n_splits, n_repeats)
                row[f"{metric}_mean"] = ci["mean"]
                row[f"{metric}_ci_low"] = ci["ci_low"]
                row[f"{metric}_ci_high"] = ci["ci_high"]
                row[f"{metric}_n_folds"] = ci["n"]
            row["majority_baseline_mean"] = group["majority_baseline_agreement"].mean()
            row["mean_black_box_errors"] = group["black_box_error_count"].mean()
            fidelity_summaries.append(row)
    return {"performance_summary": performance,
            "auc_comparisons": auc_comparisons,
            "fidelity_summary": pd.DataFrame(fidelity_summaries)}
