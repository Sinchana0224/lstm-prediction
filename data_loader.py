import yfinance as yf
import pandas as pd

def load_gold_data():
    df = yf.download("GC=F", start="2015-01-01", progress=False)

    # 🔥 HANDLE MULTI-INDEX ISSUE
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']]
    df.dropna(inplace=True)

    return df