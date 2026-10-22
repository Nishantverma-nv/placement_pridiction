import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICALS, DROPS, NUMERICS, PRE_TEST, PRE_TRAIN, PIPELINE, TARGET,
)
from src.data.load_data import load_data

# ordinal encodings: higher value = stronger profile
TIER_CATEGORIES = [["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]]


def build_preprocessor():
    numeric = [c for c in NUMERICS if c not in ("CollegeTier", "CGPA_Tier")]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=TIER_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    category_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("tier", ordinal_pipe, ["CollegeTier", "CGPA_Tier"]),
        ("cat", category_pipe, CATEGORICALS),
    ])


def run_placement_pipeline():
    df = load_data()
    y = df[TARGET].astype(int)
    X = df.drop(columns=DROPS + [TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    def transform(X):
        cols = list(preprocessor.get_feature_names_out())
        return pd.DataFrame(preprocessor.transform(X), columns=cols, index=X.index)

    Xtr, Xte = transform(X_train), transform(X_test)
    Xtr[TARGET], Xte[TARGET] = y_train, y_test

    Xtr.to_csv(PRE_TRAIN, index=False)
    Xte.to_csv(PRE_TEST, index=False)
    joblib.dump(preprocessor, PIPELINE)

    return preprocessor, Xtr, Xte, y_train, y_test


if __name__ == "__main__":
    prep, Xtr, Xte, ytr, yte = run_placement_pipeline()
    print(f"saved {PRE_TRAIN} ({Xtr.shape}) and {PRE_TEST} ({Xte.shape})")