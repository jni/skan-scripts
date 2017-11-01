# IPython log file


get_ipython().run_line_magic('run', '-i compare-skan-fiji.py')
get_ipython().run_line_magic('run', '-i compare-skan-fiji.py')
axes[0].add_collection(collections.LineCollection(coords0_skan, colors='C0'))
plt.show()
coords_cols0_fiji = ['V1 y', 'V1 x', 'V2 y', 'V2 x']
coords0_fiji = df2[coords_cols0_fiji].values.reshape((-1, 2, 2))
axes[0].add_collection(collections.LineCollection(coords0_fiji, colors='C2'))
plt.draw_idle()
fig.canvas.draw_idle()
df2.columns
coords0_fiji /= np.concatenate((spacing[1:], spacing[1:]))
coords0_fiji /= np.concatenate((spacing[1:], spacing[1:])).reshape((2, 2))
axes[0].add_collection(collections.LineCollection(coords0_fiji, colors='C2'))
axes[0].clear()
axes[0].imshow(mip0)
axes[0].imshow(mip0, cmap='magma')
axes[0].clear()
axes[0].imshow(mip0, cmap='magma')
axes[0].add_collection(collections.LineCollection(coords0_skan, colors='C0'))
coords_cols0_fiji = ['V1 x', 'V1 y', 'V2 x', 'V2 y']
coords0_fiji = (df2[coords_cols0_fiji].values.reshape((-1, 2, 2)) / spacing[0])
axes[0].add_collection(collections.LineCollection(coords0_fiji, colors='C2'))
axes[0].clear()
axes[0].imshow(mip0, cmap='magma')
axes[0].add_collection(collections.LineCollection(coords0_skan, colors='C0'))
axes[0].add_collection(collections.LineCollection(coords0_fiji, colors='C2'))
axes[0].add_collection(collections.LineCollection(coords0_fiji * spacing[1], colors='C2'))
axes[0].set_xlim(150, 450)
axes[0].set_xlim(100, 456)
axes[0].set_xlim(100, 460)
mip0.shape
axes[0].set_xlim(100, 480)
axes[0].set_ylim(350, 120)
axes[2].imshow(mip1, aspect=3, cmap='magma')
axes[1].imshow(mip2.T, aspect=3, cmap='magma')
axes[1].clear()
axes[1].imshow(mip2.T, aspect=1/3, cmap='magma')
plt.figure()
degree_image.shape
plt.imshow(np.max(degree_image, axis=0))
plt.imshow(np.max(degree_image, axis=0, cmap='magma'))
plt.imshow(np.max(degree_image, axis=0), cmap='magma')
plt.imshow(np.argmax(image3d, axis=0), cmap='magma')
plt.clf(); plt.hist(image3d.ravel(), bins=np.arange(0, 256))
plt.clf(); plt.hist(image3d.ravel(), bins=np.arange(1, 256));
image3d2 = np.copy(image3d)
image3d2[image3d < 200] = 0
plt.clf(); plt.imshow(np.argmax(image3d2, axis=0), cmap='magma')
image3d2 = np.copy(image3d)
image3d2[image3d < 100] = 0
plt.clf(); plt.imshow(np.argmax(image3d2, axis=0), cmap='magma')
fig, ax = plt.subplots(2, 2)
axes = ax.ravel()
axes[0].imshow(np.argmax(image3d2, axis=0), cmap='magma')
axes[0].set_axis_off()
axes[0].set_ylim(450, 100)
scale
spacing
200 / spacing[1]
500 / spacing[1]
axes[0].plot((300, 465), (400, 400), c='w', lw=2)
axes[0].colorbar()
plt.colorbar(np.argmax(image3d2, axis=0) * spacing[0], cax=axes[0])
fig.clf()
im = axes[0].imshow(np.argmax(image3d2, axis=0) * spacing[0], cmap='magma')
axes[0].set_ylim(450, 100)
axes[0].set_axis_off()
axes[0].plot((300, 465), (400, 400), c='w', lw=2)
plt.colorbar(im, cax=axes[0])
fig, ax = plt.subplots(2, 2)
axes = ax.ravel()
im = axes[0].imshow(np.argmax(image3d2, axis=0) * spacing[0], cmap='magma')
get_ipython().run_line_magic('pinfo', 'plt.colorbar')
axes[0].set_ylim(450, 100)
axes[0].set_axis_off()
axes[0].plot((300, 465), (400, 400), c='w', lw=2)
cbar = plt.colorbar(im, cax=axes[0])
image3d2.shape
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots(2, 2)
axes = ax.ravel()
im = axes[0].imshow(np.argmax(image3d2, axis=0) * spacing[0], cmap='magma')
axes[0].set_ylim(450, 100)
axes[0].set_axis_off()
divider3 = make_axes_locatable(axes[0])
cax = divider3.append_axes('right', size='10%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
get_ipython().run_line_magic('pinfo', 'plt.subplots_adjust')
imskel = axes[1].imshow(1 - np.max(skel1, axis=0), cmap='gray')
imskel = axes[1].imshow(np.max(skel1, axis=0), cmap='gray')
skel1.dtype
imskel = axes[1].imshow(255 - np.max(skel1, axis=0), cmap='gray')
axes[1].set_axis_off()
axes[1].set_ylim(450, 100)
axes[0].plot((300, 465), (400, 400), c='w', lw=2)
axes[1].plot((300, 465), (400, 400), c='k', lw=2)
get_ipython().run_line_magic('whos', 'bins')
type(bins)
axes[2].hist(df['branch-distance'], bins=bins, label='skan')
axes[2].clear()
bins = np.histogram(np.concatenate((df['branch-distance'], df2['Branch length'])), bins='auto')[1]
axes[2].hist(df['branch-distance'], bins=bins, label='skan');
axes[2].hist(df2['Branch length'], bins=bins, label='Fiji');
axes[2].clear()
axes[2].hist(df['branch-distance'], bins=bins, label='skan');
axes[2].hist(df2['Branch length'], bins=bins, label='Fiji', alpha=0.3);
axes[2].set_xlabel('Branch length (
axes[2].set_xlabel('Branch length (µm)')
axes[2].set_ylabel('count')
axes[2].legend()
dmx.shape
counts_points, bins_points = np.histogram(values, bins=100)
axes[3].hist(values, bins=bins)
values = dmx[np.arange(dmx.shape[0]), assignments]
axes[3].clear()
axes[3].hist(values, bins=bins_points)
counts_points, bins_points = np.histogram(values, bins=50)
axes[3].clear()
axes[3].hist(values, bins=bins_points);
axes[3].set_xlabel('Distance from skan junction to nearest\nFiji junction (µm)')
axes[3].set_xlabel('Distance from skan junction to\nnearest Fiji junction (µm)')
axes[3].set_ylabel('count')
fig.tight_layout()
get_ipython().run_line_magic('pwd', '')
fig.savefig('Fiji-skan-comparison-summary.png', dpi=600)
len(bins)
axes[3].hist(values, bins=len(bins));
np.sqrt(81 + 9 + 9)
