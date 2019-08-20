import numpy as np
import pandas as pd

url = 'http://vfacstaff.ltu.edu/lshamir/data/assym/p_all_full.csv'
file_loc = 'data.csv'
df = pd.read_csv(file_loc, nrows=10, index_col=0)
petro_df = df.filter(like='petro', axis=1)
print(petro_df.head())
col_names = pd.read_csv('col_names.csv')
print(col_names)