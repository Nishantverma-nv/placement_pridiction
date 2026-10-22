import pandas as pd

from src.config import RAW_DATA, TARGET


def load_data():
    return pd.read_csv(RAW_DATA)


def get_summary(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": TARGET,
        "nulls": int(df.isna().sum().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items()},
        "target_counts": {int(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "placed_rate": float(round(df[TARGET].mean(), 4)),
    }


if __name__ == "__main__":
    df = load_data()
    print("Summary:", get_summary(df))
    print(df.head())
