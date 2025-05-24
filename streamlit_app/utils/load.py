import pandas as pd
from pathlib import Path
import rarfile

def load_data():
    df = pd.read_csv(
            Path("../data/dailyclimate.csv.gz"),
            compression="gzip",
            dayfirst=True
    )
    return(df)
