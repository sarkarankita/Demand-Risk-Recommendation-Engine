from pathlib import Path
import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load raw access log data from CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    return pd.read_csv(path)


def preprocess_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Convert date column to datetime and sort.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    return df


def build_hourly_demand(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    Aggregate event-level data into hourly demand.
    """
    hourly = (
        df
        .set_index(date_col)
        .resample("h")
        .size()
    )
    hourly.name = "hourly_demand"
    return hourly
