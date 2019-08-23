import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
#Searching for outliers

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

#Coordinates - ensuring cw and ccw galaxies are equally distributed across the sky
#ax1 = sns.scatterplot(x='ra', y='dec', data=coordinates, hue='rotation', alpha=0.7, palette='dark', s=3)
#ax1.set_title('Coordinates')
#ax1.set_xlabel('Right ascension (RA) (degrees)')
#ax1.set_ylabel('Declination (Dec) (degrees)')
#plt.show()


#Petrosian - checking for outliers in magnitude across U & Z bands
#ax2 = sns.boxplot(x='variable', y='value', hue='rotation', 
#           data=pd.melt(petro[['rotation', 'petroMag_u', 'petroMag_z']], id_vars=['rotation']))
#ax2.set_title('Petrosian Magnitude (U & Z)')
#ax2.set_xlabel('Band (U or Z)')
#ax2.set_ylabel('Magnitude (magnitudes)')
#plt.show()


#Sky - checking for outliers in flux across U & Z bands
#ax3 = sns.boxplot(x='variable', y='value', hue='rotation', 
#           data=pd.melt(sky[['rotation', 'sky_u', 'sky_z']], id_vars=['rotation']))
#ax3.set(yscale='log')
#ax3.set_title('Sky Flux (U & Z)')
#ax3.set_xlabel('Band (U or Z)')
#ax3.set_ylabel('Flux')
#plt.show()

#Stokes - checking for outliers in polarization values across U & Z bands
#ax4 = sns.boxplot(x='variable', y='value', hue='rotation', 
#           data=pd.melt(stokes[['rotation', 'q_u', 'q_z']], id_vars=['rotation']))
#ax4.set_title('Stokes Q Parameter (U & Z)')
#ax4.set_xlabel('Band (U or Z)')
#ax4.set_ylabel('Polarization')
#plt.show()

print(stokes[abs(stokes['q_u']) > 5])

#plt.show()
















