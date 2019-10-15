import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.random import seed

# Defining helper functions that perform frequently used tasks, such as
# returning a cw/ccw portion of a df or filtering out a specific band.
def get_cw(df):
    """
    Returns the cw entries of a given dataframe.
    """
    return df[df['rotation'] == 'cw']

def get_ccw(df):
    """
    Returns the ccw entries of a given dataframe.
    """
    return df[df['rotation'] == 'ccw']

def get_band(df, band):
    """
    Returns data from a specific band of a given dataframe.
    """
    return df.filter(regex='_%s' % band)


# Loading all of the individual dataframes
df_list = []
coordinates = pd.read_csv('coordinates.csv', index_col=0)
df_list.append(coordinates)
devaucouleurs = pd.read_csv('devaucouleurs.csv', index_col=0)
df_list.append(devaucouleurs)
exponential = pd.read_csv('exponential.csv', index_col=0)
df_list.append(exponential)
extinction = pd.read_csv('extinction.csv', index_col=0)
df_list.append(extinction)
fiber = pd.read_csv('fiber.csv', index_col=0)
df_list.append(fiber)
flags = pd.read_csv('flags.csv', index_col=0)
df_list.append(flags)
isophotal = pd.read_csv('isophotal.csv', index_col=0)
df_list.append(isophotal)
m = pd.read_csv('m.csv', index_col=0)
df_list.append(m)
model = pd.read_csv('model.csv', index_col=0)
df_list.append(model)
object_info = pd.read_csv('object_info.csv', index_col=0)
df_list.append(object_info)
petro = pd.read_csv('petro.csv', index_col=0)
df_list.append(petro)
position = pd.read_csv('position.csv', index_col=0)
df_list.append(position)
prof = pd.read_csv('prof.csv', index_col=0)
df_list.append(prof)
psf = pd.read_csv('psf.csv', index_col=0)
df_list.append(psf)
signal = pd.read_csv('signal.csv', index_col=0)
df_list.append(signal)
sky = pd.read_csv('sky.csv', index_col=0)
df_list.append(sky)
stokes = pd.read_csv('stokes.csv', index_col=0)
df_list.append(stokes)
target = pd.read_csv('target.csv', index_col=0)
df_list.append(target)
texture = pd.read_csv('texture.csv', index_col=0)
df_list.append(texture)
types = pd.read_csv('types.csv', index_col=0)
df_list.append(types)



# Iterates over columns of a given df and selects columns where a t-test between
# cw and ccw entries (in any band) produce a p-value of p < 0.05. This acts as a
# crude way to perform feature selection.

significant_cols = []
bands = ['u', 'g', 'r', 'i', 'z']
for df in df_list:
    for col in df.columns:
        for band in bands:
            try:
                t_val, p_val = stats.ttest_ind(get_band(get_cw(df).dropna(), band)[col], 
                            get_band(get_ccw(df).dropna(), band)[col])
                if p_val <= 0.05:
                    print(col, t_val, p_val)
                    significant_cols.append(col)
            except:
                continue

print(significant_cols)