import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .utils import ensure_dirs

CANDIDATE_TARGETS = ['Class', 'Label', 'target', 'fraud', 'Fraud', 'is_fraud']


def detect_target(df: pd.DataFrame, preferred='auto'):
    if preferred and preferred != 'auto' and preferred in df.columns:
        return preferred
    for name in CANDIDATE_TARGETS:
        if name in df.columns:
            return name
    last = df.columns[-1]
    try:
        uniq = set(df[last].dropna().unique())
    except Exception:
        uniq = set()
    if uniq.issubset({0, 1}):
        return last
    raise ValueError("Hedef sutun bulunamadi. Lütfen config: preprocess.target içinde belirtin.")


def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {path}")
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, scale_cols=("Time", "Amount")) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    if "Amount" in scale_cols and "Amount" in df.columns:
        df['Normalized_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    if "Time" in scale_cols and "Time" in df.columns:
        df['Normalized_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    drop_cols = [c for c in scale_cols if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis=1)
    return df


def split_save(df: pd.DataFrame, target_col: str, train_csv: str, test_csv: str,
               test_size=0.2, random_state=42, stratify=True):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    ensure_dirs(os.path.dirname(train_csv), os.path.dirname(test_csv))
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return X_train, X_test, y_train, y_test


def load_processed(train_csv: str, test_csv: str, target_col: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test
