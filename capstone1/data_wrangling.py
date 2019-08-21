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


url = 'http://vfacstaff.ltu.edu/lshamir/data/assym/p_all_full.csv'
file_loc = 'data.csv'
df = pd.read_csv(file_loc, index_col=0)

#Dataset has 455 columns, so we need to group them into categories to make our analysis manageable.
col_names = pd.read_csv('col_names.csv', names=['category', 'col_name'])
col_groups = col_names.groupby('category')['col_name'].apply(list).to_dict()

#Constructing a new dataframe with multi-level columns corresponding to the categories specified above.
data_df = pd.concat([df[col_groups[key]] for key in col_groups.keys()], 
            axis=1, keys=col_groups.keys())
data_df.insert(0, 'rotation', df['rotation'])


coord_df = subset_df(data_df, 'coordinates')
sns.scatterplot(x='ra', y='dec', data=coord_df, hue='rotation', alpha=0.5, palette='dark', s=3)

plt.show()
















