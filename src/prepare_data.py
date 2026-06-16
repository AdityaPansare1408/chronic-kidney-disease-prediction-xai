# prepare_data.py
import os
import argparse
import logging
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("src/artifacts", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/shap", exist_ok=True)
    os.makedirs("outputs/shap_cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("app", exist_ok=True)

def load_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def drop_id_col(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        logger.info("Dropped 'id' column.")
    return df

def strip_whitespace_in_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def coerce_numeric_columns(df: pd.DataFrame, numeric_candidates=None) -> pd.DataFrame:
    df = df.copy()
    if numeric_candidates is None:
        candidates = []
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().astype(str).head(50).tolist()
            if any(ch.isdigit() for s in sample for ch in s):
                candidates.append(col)
    else:
        candidates = numeric_candidates

    for col in candidates:
        coerced = pd.to_numeric(df[col].replace({"?": np.nan, "na": np.nan}), errors="coerce")
        num_non_na = coerced.notna().sum()
        if num_non_na > 0:
            df[col] = coerced
            logger.info(f"Coerced column to numeric: {col} (non-NA values: {num_non_na})")
    return df

def identify_target(df: pd.DataFrame) -> str:
    possible_targets = ["classification", "class", "ckd", "target"]
    for t in possible_targets:
        if t in df.columns:
            return t
    return df.columns[-1]

def clean_target(y_series: pd.Series) -> pd.Series:
    y = y_series.astype(str).str.strip().str.lower()
    mapping = {
        "ckd": 1,
        "ckd\t": 1,
        "yes": 1,
        "notckd": 0,
        "not_ckd": 0,
        "no": 0,
        "nonckd": 0,
        "healthy": 0,
    }
    y_mapped = y.map(mapping)
    unmapped_mask = y_mapped.isna()
    if unmapped_mask.any():
        y_mapped.loc[unmapped_mask] = y.loc[unmapped_mask].apply(lambda s: 1 if "ckd" in s else 0)
    return y_mapped.astype(int)

def impute_numeric(df: pd.DataFrame, strategy: str = "median") -> Tuple[pd.DataFrame, SimpleImputer]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy=strategy)
    df_num = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols)
    df_other = df.drop(columns=num_cols)
    df_out = pd.concat([df_num, df_other.reset_index(drop=True)], axis=1)
    return df_out, imputer

def impute_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cats_map = {}
    df = df.copy()
    for col in cat_cols:
        mode = df[col].mode(dropna=True)
        if mode.shape[0] == 0:
            fill = "missing"
        else:
            fill = mode.iloc[0]
        df[col] = df[col].fillna(fill)
        df[col] = df[col].astype("category")
        cats_map[col] = list(df[col].cat.categories)
        df[col] = df[col].cat.codes
        logger.info(f"Encoded categorical column: {col} (classes: {len(cats_map[col])})")
    return df, cats_map

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler

def check_class_balance(y: pd.Series) -> None:
    logger.info("Class distribution:")
    logger.info(str(y.value_counts(normalize=False).to_dict()))
    logger.info(str(y.value_counts(normalize=True).round(3).to_dict()))

def save_processed(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                   scaler: StandardScaler, category_maps: Dict[str, object], num_imputer: SimpleImputer = None):
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    joblib.dump(scaler, "src/artifacts/scaler.joblib")
    joblib.dump(category_maps, "src/artifacts/category_maps.joblib")
    if num_imputer is not None:
        joblib.dump(num_imputer, "src/artifacts/num_imputer.joblib")
    logger.info("Saved processed files and artifacts.")

def prepare_data(input_path: str, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    ensure_dirs()

    df = load_dataset(input_path)
    df = clean_column_names(df)
    df = drop_id_col(df)
    df = strip_whitespace_in_object_cols(df)

    target_col = identify_target(df)
    logger.info(f"Identified target column: {target_col}")

    df = coerce_numeric_columns(df)

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    y = clean_target(y_raw)

    X, num_imputer = impute_numeric(X, strategy="median")

    X, category_maps = impute_categorical(X)

    X = X.select_dtypes(include=[np.number])

    check_class_balance(y)

    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    save_processed(X_train_scaled, X_test_scaled, y_train, y_test, scaler, category_maps, num_imputer)

    logger.info("Preprocessing finished successfully.")
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "category_maps": category_maps,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CKD dataset for modeling")
    parser.add_argument("--input", type=str, default="data/raw/kidney_disease.csv", help="Path to raw csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    artifacts = prepare_data(input_path=args.input, test_size=args.test_size, random_state=args.random_state)
    logger.info("Artifacts keys: %s", list(artifacts.keys()))
