# IPython log file
import os
import matplotlib.pyplot as plt
import pandas as pd
import re  # regular expressions, for text matching
import seaborn.apionly as sns

#data = pd.read_excel('/Users/jni/Dropbox/data1/malaria/adam-oli-schizont-30sec-output/skeleton.xlsx')

def infection_status(filename):
    if 'Uninf' in filename:
        return 'normal'
    else:
        return 'infected'
    
#data['infection'] = data['filename'].apply(infection_status)

def cell_number(filename):
    regex = r'.*RBC(\d+)_\d\d.tif'
    regex = re.compile(regex)
    match = regex.match(filename)
    if match is not None:
        return int(match.group(1))
    else:
        return None
    
def field_number(filename):
    regex = r'.*RBC\d+_(\d\d).tif'
    regex = re.compile(regex)
    match = regex.match(filename)
    if match is not None:
        return int(match.group(1))
    else:
        return None
    
#data['cell number'] = data['filename'].apply(cell_number)
#data['field number'] = data['filename'].apply(field_number)

# save the full data with additional columns
directory = '/Users/jni/Dropbox/data1/malaria/adam-oli-schizont-30sec-output'
#data.to_hdf(os.path.join(directory, 'skeleton-preprocess.hdf'), key='pre0')
data = pd.read_hdf(os.path.join(directory, 'skeleton-preprocess.hdf'),
                   key='pre0')
# subset the data to only have specific shape index and branch type
ridges = ((data['mean shape index'] < 0.625) &
          (data['mean shape index'] > 0.125))
j2j = data['branch-type'] == 2
datar = data.loc[ridges & j2j]
datar['branch distance (nm)'] = datar['branch-distance'] * 1e9

# Plot the full dataset boxplot
fig, ax = plt.subplots()
sns.boxplot(data=datar, x='infection', y='branch distance (nm)', ax=ax)
fig.savefig('simple-boxplot.png')

# Plot the boxplot of the image means
fig, ax = plt.subplots()
## First, group by filename and compute the means
means = datar.groupby('filename').mean().reset_index()
## This deletes non-numeric columns so we need to add them back
means['infection'] = means['filename'].apply(infection_status)
sns.boxplot(data=means, x='infection', y='branch distance (nm)', ax=ax)
fig.savefig('grouped-boxplot.png')

# plot the boxplot of the cell means
fig, ax = plt.subplots()
cellmeans = datar.groupby(('infection', 'cell number')).mean().reset_index()
sns.boxplot(x='infection', y='branch distance (nm)', hue='cell number',
            data=cellmeans, ax=ax)
ax.legend_.remove()
fig.savefig('by-cell-boxplot.png')

fig, ax = plt.subplots()
sns.stripplot(x='infection', y='branch distance (nm)', data=cellmeans,
              jitter=True, order=('normal', 'infected'), ax=ax);
ax.set_ylabel('mean branch distance by cell (nm)')
ax.set_xlabel('infection status')
fig.savefig('mean-branch-distance-by-cell.png', dpi=300)

means2 = cellmeans.groupby('infection').mean()['branch distance (nm)']
infected_mean, normal_mean = means2.loc['infected'], means2.loc['normal']
expansion_percent = (infected_mean / normal_mean - 1) * 100
print(f'infected nm: {infected_mean:.1f},\n'
      f'normal nm: {normal_mean:.1f},\n'
      f'expansion: {expansion_percent:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes[0]
_, bins = np.histogram(datar['branch distance (nm)'], bins='auto')
ax.clear()
for inf, df in datar.sort_values(by='infection', ascending=False).groupby('infection', sort=False):
    ax.hist(df['branch distance (nm)'], bins=bins, normed=True,
            alpha=0.5, label=inf)
            
ax.legend()
ax.set_xlabel('branch distance (nm)')
ax.set_ylabel('density')

ax = axes[1]
sns.stripplot(x='infection', y='branch distance (nm)', data=cellmeans,
              jitter=True, order=('normal', 'infected'), ax=ax);
ax.set_xlabel('infection status')
ax.set_ylabel('mean branch distance by cell (nm)')

fig.tight_layout()
fig.savefig('branch distance by infection status.png', dpi=300)
