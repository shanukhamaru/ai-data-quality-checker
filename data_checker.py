import pandas as pd

def basic_stats(df: pd.DataFrame):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

def missing_values(df: pd.DataFrame):
    return df.isnull().sum().to_dict()

def duplicate_rows(df: pd.DataFrame):
    return {"duplicate_row_count": df.duplicated().sum()}

def constant_columns(df: pd.DataFrame, threshold=0.98):
    constant_cols = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True).max()
        if top_freq >= threshold:
            constant_cols.append(col)
    return constant_cols
