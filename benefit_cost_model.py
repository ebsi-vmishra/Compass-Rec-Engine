#!/usr/bin/env python
# benefit_cost_model.py — group-aware split + leak guard

import argparse, json, os, sys
from dataclasses import dataclass, asdict
from typing import List, Tuple
import joblib, numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import (
    KFold, GroupKFold, GroupShuffleSplit, cross_val_score, train_test_split
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ----------------------------- utils
def ensure_outdir(path: str): os.makedirs(path, exist_ok=True)

def drop_duplicate_named_columns(df: pd.DataFrame) -> pd.DataFrame:
    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        print(f">>> Duplicate column names found (keeping first copy only): {dups}")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    assert df.columns.is_unique
    return df

def parse_list(raw: str) -> List[str]:
    if not raw: return []
    return [c.strip() for c in raw.split(",") if c.strip()]

def _assert_unique_cols(name: str, X: pd.DataFrame):
    if not X.columns.is_unique:
        dups = X.columns[X.columns.duplicated()].tolist()
        raise AssertionError(f"{name} has duplicate column names: {dups}")

def safe_intersect(cols: List[str], df_cols) -> List[str]:
    return [c for c in cols if c in df_cols]

def numeric_and_categorical_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = sorted([c for c in df.columns if is_numeric_dtype(df[c])])
    cat = sorted([c for c in df.columns if c not in num])
    if set(num) & set(cat): raise ValueError("Type overlap detected.")
    return num, cat

def drop_exact_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates()
    if len(df2) != before:
        print(f">>> Dropped {before - len(df2)} exact duplicate rows.")
    return df2

# ----------------------------- leakage guard
def leak_guard_columns(X: pd.DataFrame, y: pd.Series, target_name: str,
                       hard_patterns: List[str], corr_tol: float = 0.999,
                       equal_tol: float = 1e-9) -> List[str]:
    """
    Returns list of columns to drop because they likely leak the target.
    Heuristics:
      1) name-based: contains hard patterns or the target token
      2) value-based: perfectly (or ~perfectly) equal to target
      3) correlation-based: |corr| >= corr_tol (numeric only)
    """
    drop = set()
    lower_tokens = [p.lower() for p in (hard_patterns or [])] + [target_name.lower()]
    cols = list(X.columns)

    # 1) name-based
    for c in cols:
        lc = c.lower()
        if any(tok in lc for tok in lower_tokens):
            drop.add(c)

    # 2) value-based equality (numeric only)
    for c in cols:
        if is_numeric_dtype(X[c]):
            try:
                if np.allclose(X[c].values.astype(float), y.values.astype(float), equal_tol, equal_tol):
                    drop.add(c)
            except Exception:
                pass

    # 3) correlation-based (numeric only)
    num_cols = [c for c in cols if is_numeric_dtype(X[c]) and X[c].nunique() > 1]
    if num_cols:
        df_num = X[num_cols].copy()
        df_num[target_name] = y
        corr = df_num.corr(numeric_only=True)[target_name].drop(labels=[target_name])
        high = corr[abs(corr) >= corr_tol].index.tolist()
        drop.update(high)
        if not corr.empty:
            top = corr.abs().sort_values(ascending=False).head(12)
            print(">>> Top |corr| with target:")
            for k, v in top.items():
                print(f"    {k}: {round(float(v), 6)}")

    to_drop = sorted(drop)
    if to_drop:
        print(f">>> Leak-guard will drop {len(to_drop)} column(s): {to_drop}")
    else:
        print(">>> Leak-guard found no obvious leaking columns.")
    return to_drop

# ----------------------------- config
@dataclass
class TrainConfig:
    csv: str
    target: str
    outdir: str
    test_size: float = 0.2
    random_state: int = 42
    cv: int = 5
    model: str = "ridge"          # ridge | elasticnet
    alpha: float = 1.0
    l1_ratio: float = 0.5
    exclude: List[str] = None
    id_like_defaults: List[str] = None
    error_score_raise: bool = False
    leak_guard: bool = True
    leak_name_patterns: List[str] = None
    group_split_cols: List[str] = None  # columns defining a group (e.g., PlanId,TierOptionId,Year)

@dataclass
class Metrics:
    holdout_rmse: float
    holdout_mae: float
    holdout_r2: float
    cv_rmse_mean: float
    cv_rmse_std: float

# ----------------------------- model
def build_estimator(cfg: TrainConfig):
    return ElasticNet(alpha=cfg.alpha, l1_ratio=cfg.l1_ratio, random_state=cfg.random_state) \
        if cfg.model.lower() == "elasticnet" else Ridge(alpha=cfg.alpha, random_state=cfg.random_state)

def build_preprocessor(X_train: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_safe = safe_intersect(numeric_cols, X_train.columns)
    cat_safe = safe_intersect(categorical_cols, X_train.columns)
    print(f">>> Using {len(num_safe)} numeric columns and {len(cat_safe)} categorical columns.")
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_safe),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_safe),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )

def evaluate(model: Pipeline, X_train, y_train, X_test, y_test,
             cv_splits: int, groups_cv=None, error_score_raise=False) -> Tuple[Metrics, np.ndarray]:
    if groups_cv is not None:
        cv = GroupKFold(n_splits=cv_splits)
        cv_iter = cv.split(X_train, y_train, groups=groups_cv[:len(X_train)])
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        cv_iter = cv.split(X_train, y_train)

    err_flag = "raise" if error_score_raise else np.nan
    cv_scores = -cross_val_score(model, X_train, y_train,
                                 cv=cv_iter,
                                 scoring="neg_root_mean_squared_error",
                                 error_score=err_flag)
    metrics_cv_mean = float(np.mean(cv_scores))
    metrics_cv_std = float(np.std(cv_scores))

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    holdout_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    holdout_mae = float(mean_absolute_error(y_test, preds))
    holdout_r2 = float(r2_score(y_test, preds))
    return Metrics(holdout_rmse, holdout_mae, holdout_r2, metrics_cv_mean, metrics_cv_std), preds

# ----------------------------- main
def main():
    p = argparse.ArgumentParser(description="Benefit cost model with group-aware split and leak guard.")
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--model", choices=["ridge", "elasticnet"], default="ridge")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--l1-ratio", type=float, dest="l1_ratio", default=0.5)
    p.add_argument("--exclude", type=str, default="")
    p.add_argument("--error-score-raise", action="store_true")
    p.add_argument("--no-leak-guard", action="store_true",
                   help="Disable automatic leakage detection/dropping.")
    p.add_argument("--group-split-cols", type=str, default="",
                   help="Comma-separated columns to group by for split/CV (e.g., 'PlanId,TierOptionId,Year').")

    args = p.parse_args()

    cfg = TrainConfig(
        csv=args.csv, target=args.target, outdir=args.outdir,
        test_size=args.test_size, random_state=args.random_state, cv=args.cv,
        model=args.model, alpha=args.alpha, l1_ratio=args.l1_ratio,
        exclude=parse_list(args.exclude),
        id_like_defaults=[
            "NetworkInternalCode","ClientInternalCode","BenefitCategoryId",
            "BenefitId","PlanId","TierOptionId","CountryName","StateProvince"
        ],
        error_score_raise=args.error_score_raise,
        leak_guard=(not args.no_leak_guard),
        leak_name_patterns=["premium","prem","employee_premium","er_contrib","employer","cost","monthly_cost"],
        group_split_cols=parse_list(args.group_split_cols)
    )

    ensure_outdir(cfg.outdir)
    df = pd.read_csv(cfg.csv)
    print(f">>> Loaded {len(df)} rows, {df.shape[1]} columns from {cfg.csv}")
    df = drop_duplicate_named_columns(df)
    df = drop_exact_duplicate_rows(df)

    if cfg.target not in df.columns:
        raise ValueError(f"Target '{cfg.target}' not found. Available: {list(df.columns)}")

    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])

    # Exclusions (ids + user provided)
    to_exclude = set((cfg.exclude or [])) | set(cfg.id_like_defaults or [])
    to_exclude.discard(cfg.target)
    to_exclude = [c for c in to_exclude if c in X.columns]
    if to_exclude:
        print(f">>> Excluding {len(to_exclude)} columns from features: {to_exclude}")
        X = X.drop(columns=to_exclude)

    # Leak guard
    if cfg.leak_guard:
        leaks = leak_guard_columns(X, y, cfg.target, cfg.leak_name_patterns)
        if leaks:
            X = X.drop(columns=leaks)

    _assert_unique_cols("X (post-exclude/leakguard)", X)

    # Split groups (based on original df columns, not only X)
    groups = None
    if cfg.group_split_cols:
        missing = [c for c in cfg.group_split_cols if c not in df.columns]
        if missing:
            print(f">>> WARNING: group-split columns missing in data: {missing}")
        if any(c in df.columns for c in cfg.group_split_cols):
            groups = df[cfg.group_split_cols].astype(str).agg("§".join, axis=1).values
            print(f">>> Grouping by {cfg.group_split_cols} for split and CV.")

    # Group-aware split if groups provided
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups[train_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, shuffle=True
        )
        groups_train = None

    _assert_unique_cols("X_train", X_train)
    _assert_unique_cols("X_test", X_test)

    num_cols, cat_cols = numeric_and_categorical_columns(X)
    preproc = build_preprocessor(X_train, num_cols, cat_cols)
    estimator = build_estimator(cfg)
    pipe = Pipeline([("preprocess", preproc), ("estimator", estimator)])

    metrics, preds = evaluate(pipe, X_train, y_train, X_test, y_test,
                              cv_splits=cfg.cv, groups_cv=groups_train,
                              error_score_raise=cfg.error_score_raise)

    print(">>> Holdout RMSE:", round(metrics.holdout_rmse, 4))
    print(">>> Holdout MAE :", round(metrics.holdout_mae, 4))
    print(">>> Holdout R2  :", round(metrics.holdout_r2, 4))
    print(">>> CV RMSE mean ± std:", round(metrics.cv_rmse_mean, 4), "±", round(metrics.cv_rmse_std, 4))

    # Save artifacts
    with open(os.path.join(cfg.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    feat_payload = {
        "numeric_columns": safe_intersect(num_cols, X_train.columns),
        "categorical_columns": safe_intersect(cat_cols, X_train.columns),
        "excluded_columns": to_exclude,
        "leak_guard_dropped": [c for c in (set(num_cols+cat_cols) - set(X.columns)) if c not in to_exclude],
        "all_input_columns": list(X.columns),
        "training_columns": list(X_train.columns),
        "sklearn_version": __import__("sklearn").__version__,
    }
    with open(os.path.join(cfg.outdir, "feature_lists.json"), "w", encoding="utf-8") as f:
        json.dump(feat_payload, f, indent=2)

    joblib.dump(pipe, os.path.join(cfg.outdir, "model.joblib"))

    pred_df = X_test.copy()
    pred_df[cfg.target] = y_test.values
    pred_df["prediction"] = preds
    pred_df["residual"] = pred_df["prediction"] - pred_df[cfg.target]
    pred_df.to_csv(os.path.join(cfg.outdir, "predictions_holdout.csv"), index=False)

    print(f">>> Artifacts written to: {cfg.outdir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        sys.exit(1)
