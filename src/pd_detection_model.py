"""
Parkinson's disease detection model.

Converted from `notebooks/PD_Detection_Model.ipynb` (originally a Google Colab
notebook) into a self-contained, VS Code / command-line friendly script.

What it does
------------
  1. Load and clean the extracted smile-feature table (Extracted_feautures2.csv).
  2. Balance the classes by random under-sampling the majority class.
  3. Split into train / test (stratified).
  4. Feature selection + an ablation study across:
       * feature-selection methods: Logistic Regression, Boosted-RFE, Boosted-RFA
       * scalers:                    MinMax, Standard
       * classifiers:                SVM, AdaBoost, HistGradientBoosting,
                                     XGBoost, RandomForest
       * number of features:         configurable list
     scored by mean AUROC over stratified K-fold CV.
  5. Correlation pruning of the best feature set (drop |r| > 0.9, keep the
     higher-ranked feature).
  6. SVM hyper-parameter tuning with Optuna (a local, offline replacement for
     the original Weights & Biases Bayesian sweep).
  7. Final training on the training split and evaluation on the held-out test
     set, saving metrics, plots and the fitted model.

The original notebook depended on Google Colab (Drive mounting) and Weights &
Biases (cloud sweeps + login). Those have been removed so the script runs fully
offline. Set --n-trials 0 to skip Optuna and evaluate the SVM with default
hyper-parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")  # headless-safe; overridden by --show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Optional dependencies (shap / xgboost / optuna). We degrade gracefully so the
# core pipeline still runs if a heavy dependency is missing.
# --------------------------------------------------------------------------- #

try:
    import shap
    HAVE_SHAP = True
except Exception:  # pragma: no cover
    HAVE_SHAP = False

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False

try:
    import optuna
    HAVE_OPTUNA = True
except Exception:  # pragma: no cover
    HAVE_OPTUNA = False


def make_scaler(name: str):
    return StandardScaler() if name == "Standard" else MinMaxScaler()


# --------------------------------------------------------------------------- #
# 1. Data loading and cleaning
# --------------------------------------------------------------------------- #

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["PD"])

    feature_cols = [c for c in df.columns if c not in ["PD", "clip"]]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df["PD"].astype(int)
    print(f"Loaded {csv_path}")
    print(f"  dataset shape: {X.shape}, target distribution: {y.value_counts().to_dict()}")
    return X, y


def balance_by_undersampling(X, y, rng):
    """Random under-sample every class down to the smallest class count."""
    least = y.value_counts().min()
    idx = []
    for label in y.unique():
        class_idx = y[y == label].index
        if len(class_idx) > least:
            idx.extend(rng.choice(class_idx, size=least, replace=False))
        else:
            idx.extend(class_idx)
    Xb, yb = X.loc[idx], y.loc[idx]
    print(f"  balanced shape: {Xb.shape}, target distribution: {yb.value_counts().to_dict()}")
    return Xb, yb


# --------------------------------------------------------------------------- #
# 2. Feature-selection methods
# --------------------------------------------------------------------------- #

def logistic_regression_feature_ranking(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_scaled, y)
    return pd.DataFrame({
        "feature": X.columns,
        "coefficient": lr.coef_[0],
        "abs_coefficient": np.abs(lr.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _shap_importance(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values.abs.mean(0).values


def boosted_rfe_feature_ranking(X, y, n_features_to_select=None):
    """Recursive feature elimination driven by SHAP importances."""
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2

    remaining = list(X.columns)
    while len(remaining) > n_features_to_select:
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        model.fit(X[remaining], y)
        importance = _shap_importance(model, X[remaining])
        scores = dict(zip(remaining, importance))
        remaining.remove(min(scores, key=scores.get))

    ranking_df = pd.DataFrame({
        "feature": remaining,
        "ranking": range(1, len(remaining) + 1),
        "selected": [True] * len(remaining),
    })
    return ranking_df, pd.Index(remaining)


def boosted_rfa_feature_ranking(X, y, n_features_to_select=None):
    """Recursive feature addition: add SHAP-ranked features while AUROC improves."""
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2

    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=100)
    model.fit(X, y)
    importance = _shap_importance(model, X)
    ranked = pd.Series(importance, index=X.columns).sort_values(ascending=False)

    selected, best_features, best_score = [], [], 0.0
    for feature in ranked.index:
        selected.append(feature)
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=100)
        model.fit(X[selected], y)
        auc = roc_auc_score(y, model.predict_proba(X[selected])[:, 1])
        if auc >= best_score:
            best_score, best_features = auc, deepcopy(selected)
        if len(selected) >= n_features_to_select:
            break

    return pd.DataFrame({
        "feature": best_features,
        "ranking": range(1, len(best_features) + 1),
        "selected": [True] * len(best_features),
    })


def available_fs_methods():
    methods = {"Logistic_Regression": logistic_regression_feature_ranking}
    if HAVE_SHAP:
        methods["BoostRFE"] = boosted_rfe_feature_ranking
        methods["BoostRFA"] = boosted_rfa_feature_ranking
    else:
        print("  [warn] shap not installed -> skipping BoostRFE / BoostRFA methods.")
    return methods


def available_classifiers():
    clfs = {
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "HistBoost": HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=100),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    if HAVE_XGB:
        clfs["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE)
    else:
        print("  [warn] xgboost not installed -> skipping XGBoost classifier.")
    return clfs


# --------------------------------------------------------------------------- #
# 3. Ablation study
# --------------------------------------------------------------------------- #

def select_features(fs_name, fs_method, X_fold, y_fold, n_features):
    """Return the top-n feature list for a fold, per the selection method."""
    if fs_name == "BoostRFE":
        _, feats = fs_method(X_fold, y_fold, n_features)
        return list(feats)
    if fs_name == "BoostRFA":
        return fs_method(X_fold, y_fold, n_features)["feature"].tolist()
    ranking = fs_method(X_fold, y_fold)
    return ranking.head(n_features)["feature"].tolist()


def ablation_study(X, y, n_features_list, n_splits, verbose=False):
    classifiers = available_classifiers()
    scalers = {"MinMax": MinMaxScaler, "Standard": StandardScaler}
    fs_methods = available_fs_methods()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for n_features in n_features_list:
        for fs_name, fs_method in fs_methods.items():
            for scaler_name, scaler_class in scalers.items():
                for clf_name, clf in classifiers.items():
                    print(f"  Evaluating {clf_name:12} | FS={fs_name:19} | "
                          f"Scaler={scaler_name:8} | top-{n_features}")
                    fold_scores, fold_feature_sets = [], []
                    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
                        X_tr, X_va = X.iloc[tr], X.iloc[va]
                        y_tr, y_va = y.iloc[tr], y.iloc[va]

                        feats = select_features(fs_name, fs_method, X_tr, y_tr, n_features)
                        scaler = scaler_class()
                        X_tr_s = scaler.fit_transform(X_tr[feats])
                        X_va_s = scaler.transform(X_va[feats])

                        clf.fit(X_tr_s, y_tr)
                        score = roc_auc_score(y_va, clf.predict_proba(X_va_s)[:, 1])
                        fold_scores.append(score)
                        fold_feature_sets.append(feats)
                        if verbose:
                            print(f"      fold {fold}/{n_splits} AUROC {score:.4f}")

                    best_fold = int(np.argmax(fold_scores))
                    results.append({
                        "fs_method": fs_name,
                        "scaler": scaler_name,
                        "classifier": clf_name,
                        "n_features": n_features,
                        "features": fold_feature_sets[best_fold],
                        "auc_mean": float(np.mean(fold_scores)),
                        "auc_std": float(np.std(fold_scores)),
                    })
    return pd.DataFrame(results)


def plot_ablation(ablation_df, out_dir):
    g = sns.FacetGrid(
        ablation_df, col="fs_method",
        col_order=sorted(ablation_df["fs_method"].unique()),
        height=4, aspect=1.5, sharey=True,
    )
    g.map_dataframe(sns.lineplot, x="n_features", y="auc_mean",
                    hue="classifier", marker="o")
    g.add_legend()
    g.set_axis_labels("Number of Features", "Mean AUROC")
    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle("AUROC vs. Number of Features per Feature-Selection Method")
    path = os.path.join(out_dir, "ablation_study.png")
    g.savefig(path, dpi=200)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# 4. Correlation-based pruning
# --------------------------------------------------------------------------- #

def recompute_ranking(fs_method, X_train, y_train, top_features, n_features):
    """Rebuild a rank lookup (feature -> rank) restricted to `top_features`."""
    if fs_method == "BoostRFE" and HAVE_SHAP:
        _, feats = boosted_rfe_feature_ranking(X_train, y_train, n_features)
        ranking = pd.DataFrame({"feature": list(feats)})
    elif fs_method == "BoostRFA" and HAVE_SHAP:
        ranking = boosted_rfa_feature_ranking(X_train, y_train, n_features)
    else:
        ranking = logistic_regression_feature_ranking(X_train, y_train)
    ranking = ranking.reset_index(drop=True)
    ranking["rank"] = ranking.index
    ranking = ranking[ranking["feature"].isin(top_features)].reset_index(drop=True)
    return ranking


def prune_correlated(X_train, top_features, ranking, threshold=0.9):
    corr = X_train[top_features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    rank_of = dict(zip(ranking["feature"], ranking["rank"]))
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and val > threshold:
                # drop the lower-ranked (larger rank number) of the pair
                if rank_of.get(col, 1e9) > rank_of.get(row, 1e9):
                    to_drop.add(col)
                else:
                    to_drop.add(row)
    final = [f for f in top_features if f not in to_drop]
    return final, to_drop


def plot_correlation(X, features, title, path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(X[features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 5. SVM tuning (Optuna) - offline replacement for the W&B Bayesian sweep
# --------------------------------------------------------------------------- #

def tune_svm(X_train, y_train, features, scaler_name, n_trials, n_splits):
    """Bayesian search over (C, gamma, kernel) maximising mean CV val-AUROC."""
    default = {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
    if n_trials <= 0 or not HAVE_OPTUNA:
        if n_trials > 0 and not HAVE_OPTUNA:
            print("  [warn] optuna not installed -> using default SVM hyper-parameters.")
        return default

    X_sel = X_train[features]
    Scaler = StandardScaler if scaler_name == "Standard" else MinMaxScaler
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
        scores = []
        for tr, va in skf.split(X_sel, y_train):
            pipe = Pipeline([
                ("scaler", Scaler()),
                ("svc", SVC(C=C, gamma=gamma, kernel=kernel,
                            probability=True, random_state=RANDOM_STATE)),
            ])
            pipe.fit(X_sel.iloc[tr], y_train.iloc[tr])
            prob = pipe.predict_proba(X_sel.iloc[va])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[va], prob))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  best CV val-AUROC: {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")
    return study.best_params


# --------------------------------------------------------------------------- #
# 6. Final training and evaluation
# --------------------------------------------------------------------------- #

def evaluate_final(X_train, X_test, y_train, y_test, features, scaler_name,
                   svm_params, out_dir):
    Scaler = StandardScaler if scaler_name == "Standard" else MinMaxScaler
    pipe = Pipeline([
        ("scaler", Scaler()),
        ("svc", SVC(probability=True, random_state=RANDOM_STATE, **svm_params)),
    ])
    pipe.fit(X_train[features], y_train)

    y_pred = pipe.predict(X_test[features])
    y_prob = pipe.predict_proba(X_test[features])[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "auroc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "sensitivity_recall": recall_score(y_test, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "ppv_precision": precision_score(y_test, y_pred, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) else 0.0,
    }

    print("\nFINAL SVM TEST-SET PERFORMANCE")
    for k, v in metrics.items():
        print(f"  {k:20}: {v:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No PD", "PD"], zero_division=0))

    # Plots: confusion matrix, ROC, precision-recall
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
                cmap="Blues", ax=ax[0])
    ax[0].set_title("Confusion Matrix")
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax[1].plot(fpr, tpr, lw=2, label=f"AUROC = {metrics['auroc']:.3f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax[1].set_title("ROC Curve"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax[2].plot(rec, prec, lw=2, label="PR Curve")
    ax[2].set_title("Precision-Recall Curve")
    ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].legend()

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "final_evaluation.png")
    plt.savefig(fig_path, dpi=200)
    print(f"\nsaved {fig_path}")
    return pipe, metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Parkinson's smile-detection SVM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default="data/Extracted_feautures2.csv",
                        help="Path to the extracted-features CSV.")
    parser.add_argument("--output-dir", default="models",
                        help="Directory for the saved model, metrics and plots.")
    parser.add_argument("--n-features-list", type=int, nargs="+",
                        default=[10, 15, 20, 25, 30],
                        help="Candidate feature-set sizes for the ablation study.")
    parser.add_argument("--cv-folds", type=int, default=10,
                        help="Stratified K-fold splits for the ablation study.")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials for SVM tuning (0 = use defaults).")
    parser.add_argument("--tune-folds", type=int, default=5,
                        help="CV folds used inside Optuna SVM tuning.")
    parser.add_argument("--quick", action="store_true",
                        help="Small/fast run for a smoke test "
                             "(few feature sizes, folds and trials).")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively instead of only saving.")
    args = parser.parse_args(argv)

    if args.show:
        matplotlib.use("TkAgg", force=True)

    if args.quick:
        args.n_features_list = [10, 15]
        args.cv_folds = 3
        args.n_trials = min(args.n_trials, 10)
        args.tune_folds = 3

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(RANDOM_STATE)

    # 1-3. Load, balance, split
    print("== 1. Load & clean ==")
    X, y = load_data(args.csv)
    print("== 2. Balance classes ==")
    Xb, yb = balance_by_undersampling(X, y, rng)
    print("== 3. Train/test split ==")
    X_train, X_test, y_train, y_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=RANDOM_STATE, stratify=yb)
    print(f"  train {X_train.shape}, test {X_test.shape}")

    # 4. Ablation study
    print("\n== 4. Ablation study ==")
    ablation_df = ablation_study(X_train, y_train, args.n_features_list, args.cv_folds)
    ablation_df.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)
    plot_ablation(ablation_df, args.output_dir)

    best = ablation_df.loc[ablation_df["auc_mean"].idxmax()]
    print("\nBest configuration from ablation:")
    print(f"  classifier={best['classifier']}, fs={best['fs_method']}, "
          f"scaler={best['scaler']}, n_features={best['n_features']}, "
          f"auc_mean={best['auc_mean']:.4f}")
    top_features = list(best["features"])

    # 5. Correlation pruning
    print("\n== 5. Correlation pruning ==")
    ranking = recompute_ranking(best["fs_method"], X_train, y_train,
                                top_features, int(best["n_features"]))
    plot_correlation(X_train, top_features, "Correlation (Before Pruning)",
                     os.path.join(args.output_dir, "corr_before_pruning.png"))
    final_features, dropped = prune_correlated(X_train, top_features, ranking)
    print(f"  removed {len(dropped)} correlated feature(s): {sorted(dropped)}")
    print(f"  final feature set ({len(final_features)}): {final_features}")
    if final_features:
        plot_correlation(X_train, final_features, "Correlation (After Pruning)",
                         os.path.join(args.output_dir, "corr_after_pruning.png"))

    # 6. SVM tuning (offline Optuna sweep)
    print("\n== 6. SVM hyper-parameter tuning (Optuna) ==")
    svm_params = tune_svm(X_train, y_train, final_features, best["scaler"],
                          args.n_trials, args.tune_folds)

    # 7. Final training + evaluation
    print("\n== 7. Final training & evaluation ==")
    model, metrics = evaluate_final(
        X_train, X_test, y_train, y_test,
        final_features, best["scaler"], svm_params, args.output_dir)

    # Save model bundle + metrics
    import joblib
    model_path = os.path.join(args.output_dir, "pd_svm_model.joblib")
    joblib.dump({
        "model": model,                       # full sklearn Pipeline (scaler + SVM)
        "features": final_features,
        "scaler": best["scaler"],
        "svm_params": svm_params,
    }, model_path)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump({
            "config": {
                "classifier": str(best["classifier"]),
                "fs_method": str(best["fs_method"]),
                "scaler": str(best["scaler"]),
                "n_features": int(best["n_features"]),
                "final_features": final_features,
                "svm_params": svm_params,
            },
            "test_metrics": metrics,
        }, fh, indent=2)

    print(f"\nSaved model   -> {model_path}")
    print(f"Saved metrics -> {os.path.join(args.output_dir, 'metrics.json')}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
