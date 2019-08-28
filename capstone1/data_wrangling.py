import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
import seaborn as sns
sns.set()

def subset_df(df, group):
    '''
    Method for creating a new dataframe for a specified group. 
    Preserves the 'rotation' column from the original dataframe.
    '''
    new_df = df[['rotation', group]].droplevel(0, axis=1)
    new_df.rename(columns={new_df.columns[0]:'rotation'}, inplace=True)
    return new_df


URL = 'http://vfacstaff.ltu.edu/lshamir/data/assym/p_all_full.csv'
df = pd.read_csv(URL)
#FILE_LOC = 'data.csv
#df = pd.read_csv(FILE_LOC, index_col=0)

#Dataset has 455 columns, so we need to group them into categories to make our analysis manageable.
col_names = pd.read_csv('col_names.csv', names=['category', 'col_name'])
col_groups = col_names.groupby('category')['col_name'].apply(list).to_dict()

#Constructing a new dataframe with multi-level columns corresponding to the categories specified above.
data_df = pd.concat([df[col_groups[key]] for key in col_groups.keys()], 
            axis=1, keys=col_groups.keys())
data_df.insert(0, 'rotation', df['rotation'])

#Creating individual dataframes for each category to perform preliminary analysis.
coordinates = subset_df(data_df, 'coordinates')
devaucouleurs = subset_df(data_df, 'devaucouleurs')
exponential = subset_df(data_df, 'exponential')
extinction = subset_df(data_df, 'extinction')
fiber = subset_df(data_df, 'fiber')
flags = subset_df(data_df, 'flags')
isophotal = subset_df(data_df, 'isophotal')
m = subset_df(data_df, 'm')
model = subset_df(data_df, 'model')
object_info = subset_df(data_df, 'object_info')
petro = subset_df(data_df, 'petro')
position = subset_df(data_df, 'position')
prof = subset_df(data_df, 'prof')
psf = subset_df(data_df, 'psf')
signal = subset_df(data_df, 'signal')
sky = subset_df(data_df, 'sky')
stokes = subset_df(data_df, 'stokes')
target = subset_df(data_df, 'target')
texture = subset_df(data_df, 'texture')
types = subset_df(data_df, 'types')

#Checking to make sure the only 'object' data types in the dataframe are in the 'rotation' column.
assert data_df.select_dtypes(['object']).equals(data_df[['rotation']])

#Checking to make sure there are no None/nan/empty string values in dataframe, and that
#none of the columns have a data type of 'object' (except the 'rotation' column).
for category, column in list(data_df):
    assert data_df[data_df.loc[:, (category, column)] == None].empty
    assert data_df[data_df.loc[:, (category, column)] == np.nan].empty
    assert data_df[data_df.loc[:, (category, column)] == ''].empty
    if category != 'rotation':
        assert data_df.loc[:, (category, column)].dtype != 'object'
    
#Searching for outliers and suspicious patterns. In this case, checking the Stokes Q value
#across U & Z bands
sns.boxplot(x='variable', y='value', hue='rotation', 
           data=pd.melt(stokes[['rotation', 'q_u', 'q_z']], id_vars=['rotation']))
plt.title('Stokes Q Parameter (U & Z)')
plt.xlabel('Band (U or Z)')
plt.ylabel('Polarization')
plt.text(x=-0.04, y=-8.2, s='outlier')
plt.text(x=-0.04, y=7.5, s='outlier')
plt.tick_params(labelsize=8)
plt.tight_layout()
plt.show()

#Removing the two outliers that were identified from the main dataset.
max_outlier = data_df['stokes']['q_u'] == data_df['stokes']['q_u'].max()
min_outlier = data_df['stokes']['q_u'] == data_df['stokes']['q_u'].min()
data_df.drop(data_df[max_outlier | min_outlier].index, inplace=True)

#Updating individual dataframes.
coordinates = subset_df(data_df, 'coordinates')
devaucouleurs = subset_df(data_df, 'devaucouleurs')
exponential = subset_df(data_df, 'exponential')
extinction = subset_df(data_df, 'extinction')
fiber = subset_df(data_df, 'fiber')
flags = subset_df(data_df, 'flags')
isophotal = subset_df(data_df, 'isophotal')
m = subset_df(data_df, 'm')
model = subset_df(data_df, 'model')
object_info = subset_df(data_df, 'object_info')
petro = subset_df(data_df, 'petro')
position = subset_df(data_df, 'position')
prof = subset_df(data_df, 'prof')
psf = subset_df(data_df, 'psf')
signal = subset_df(data_df, 'signal')
sky = subset_df(data_df, 'sky')
stokes = subset_df(data_df, 'stokes')
target = subset_df(data_df, 'target')
texture = subset_df(data_df, 'texture')
types = subset_df(data_df, 'types')

#Our data is now clean and ready for analysis!













