# IPython log file
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from skimage import io
from skan import csr

#current_directory = '/Users/jni/projects/skan-scripts'
skel1 = io.imread('OP_1_Rendered_Paths_thinned.tif')
spacing = [3.033534 * 3, 3.033534, 3.033534]
spacing = np.asarray(spacing)
df = csr.summarise(skel1.astype(bool), spacing=spacing)
df2 = pd.read_excel('OP_1-Branch-information.xlsx')
dfs = df.sort_values(by='branch-distance', ascending=False)
df2s = df2.sort_values(by='Branch length', ascending=False)
bins = np.histogram(np.concatenate((df['branch-distance'],
                                    df2['Branch length'])),
                    bins='auto')[1]
                    
fig, ax = plt.subplots()
ax.hist(df['branch-distance'], bins=bins, label='skan');
ax.hist(df2['Branch length'], bins=bins, label='Fiji', alpha=0.3);
ax.set_xlabel('branch length (µm)')
ax.set_ylabel('count')
fig.savefig('OP1-branch-length-histogram-new.png')

coords0 = df[['coord-0-0', 'coord-0-1', 'coord-0-2']].values
coords1 = df[['coord-1-0', 'coord-1-1', 'coord-1-2']].values
dm = distance_matrix(coords0, coords1)
all_points_skan = np.concatenate([coords0, coords1[np.where(np.min(dm, axis=0) > 1e-6)[0]]], axis=0)
coords0fj = df2[['V1 z', 'V1 y', 'V1 x']].values
coords1fj = df2[['V2 z', 'V2 y', 'V2 x']].values
dmfj = distance_matrix(coords0fj, coords1fj)
all_points_fiji = np.concatenate([coords0fj, coords1fj[np.where(np.min(dmfj, axis=0) > 1e-6)[0]]], axis=0)
dmx = distance_matrix(all_points_skan, all_points_fiji)
assignments = np.argmin(dmx, axis=1)
values = dmx[np.arange(dmx.shape[0]), assignments]
counts, bins = np.histogram(values, bins=100)
fig, ax = plt.subplots()
ax.hist(values, bins=bins);
ax.set_xlabel('distance from skan point to nearest Fiji point (µm)')
ax.set_ylabel('count')
fig.savefig('OP1-point-distance-histogram-new.png')
# above: clear difference in the branch point location due to
# cluster of junction points
