import pandas as pd
import numpy as np

def load_data():
    df=pd.read_csv('../data/dailyclimate.csv')
    return(df)
    
