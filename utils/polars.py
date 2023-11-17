from pathlib import Path
import polars as pl

df_path = Path(__file__).parent.parent / 'data' / 'phrases.csv'

df = pl.read_csv(df_path)

df.head(10)
