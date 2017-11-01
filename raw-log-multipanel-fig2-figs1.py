# IPython log file


import h5py as h5
f = h5.open('skeleton-predict.ilp')
f = h5.File('skeleton-predict.ilp')
f.groups
f.attrs
f.items()
f.keys()
list(f.keys())
f['FeatureSelections']
list(f['FeatureSelections'])
len(list(f['Input Data']))
list(f['Input Data'])
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('ls', 'adam-gametocytes-gh9-catherine | less')
dir1 = 'adam-gametocytes-gh9-catherine'
dir2 = 'adam-mix4'
fns1 = [os.path.join(dir1, fn) for fn in os.path.listdir(dir1)
        if 'ontrol' in fn and not fn.endswith('01.tif')]
        
fns1 = [os.path.join(dir1, fn) for fn in os.listdir(dir1)
        if 'ontrol' in fn and not fn.endswith('01.tif')]
        
len(fns1)
fns1[:10]
fns1 = [os.path.join(dir1, fn) for fn in os.listdir(dir1)
        if 'ontrol' in fn and not fn.endswith('01.tif')
        and fn.endswith('.tif')]
        
len(fns1)
fns2 = [os.path.join(dir2, fn) for fn in os.listdir(dir1)
        if fn.endswith('.tif') and not fn.endswith('01.tif')]
        
len(fns2)
(221 + 141) * (1536 * 1024) / 1e6
im0 = iio.imread(fns1[0], format='fei')
im0.shape
dataset = np.empty((len(fns1) + len(fns2),) + im0.shape,
                   dtype=np.uint8)
                   
for i, fn in enumerate(fns1 + fns2):
    dataset[i] = iio.imread(fn, format='fei')
    
fns2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2)
        if fn.endswith('.tif') and not fn.endswith('01.tif')]
        
len(fns2)
dataset = np.empty((len(fns1) + len(fns2),) + im0.shape,
                   dtype=np.uint8)
                   
for i, fn in enumerate(fns1 + fns2):
    dataset[i] = iio.imread(fn, format='fei')
    
fns2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2)
        if fn.endswith('.tif') and not fn.endswith('01.tif')
        and not fn.endswith('(200k).tif')]
        
len(fns2)
dataset = np.empty((len(fns1) + len(fns2),) + im0.shape,
                   dtype=np.uint8)
                   
for i, fn in enumerate(fns1 + fns2):
    dataset[i] = iio.imread(fn, format='fei')
    
from gala import imio
imio.write_h5_stack(dataset, 'learning-images.h5', compression='lzf')
plt.imshow(im0)
plt.imshow(dataset[5])
np.max(dataset[5])
im5 = iio.imread(fns1[5], format='fei')
fns1[5]
plt.imshow(im5)
im5.max()
from skimage import img_as_ubyte
plt.imshow(img_as_ubyte(im5))
dataset = np.empty((len(fns1) + len(fns2),) + im0.shape,
                   dtype=np.uint8)
                   
for i, fn in enumerate(fns1 + fns2):
    dataset[i] = img_as_ubyte(iio.imread(fn, format='fei'))
    
imio.write_h5_stack(dataset, 'learning-images.h5', compression='lzf')
fns1[120]
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('ls', '-l')
get_ipython().run_line_magic('ls', '-l')
get_ipython().run_line_magic('ls', '-l')
get_ipython().run_line_magic('pwd', '')
def write_fei_mock(image, scale, fn):
    image2 = np.pad(image, ((0, 70), (0, 0)), cval=0)
    scale_hdr = b'\r\nDate=13/10/2017\r\n[Scan]\r\nPixelHeight='
    scale_str = scale_hdr + str(scale).encode('ascii') + b'\r\n'
    iio.imsave(fn, image2)
    with open(fn, 'ab') as fout:
        fout.write(scale_str)
        
import numpy as np
import imageio as iio
fprob = h5py.File('learning-images_Probabilities.h5')
import h5py
fprob = h5py.File('learning-images_Probabilities.h5')
list(fprob)
list(fprob['exported_data'])
list(fprob['exported_data'].keys())
fprob['exported_data'].shape
probs = np.array(fprob['exported_data']).squeeze()
def get_scale(fn):
    im = iio.imread(fn, format='fei')
    scale = float(im.meta['Scan']['PixelHeight'])
    return scale
for original_fn, prob in zip(fns1 + fns2, probs):
    scale = get_scale(original_fn)
    output_fn = original_fn[:-4] + '-probabilities.tif'
    write_fei_mock(prob, scale, output_fn)
    
def write_fei_mock(image, scale, fn):
    image2 = np.pad(image, ((0, 70), (0, 0)), mode='constant', cval=0)
    scale_hdr = b'\r\nDate=13/10/2017\r\n[Scan]\r\nPixelHeight='
    scale_str = scale_hdr + str(scale).encode('ascii') + b'\r\n'
    iio.imsave(fn, image2)
    with open(fn, 'ab') as fout:
        fout.write(scale_str)
        
for original_fn, prob in zip(fns1 + fns2, probs):
    scale = get_scale(original_fn)
    output_fn = original_fn[:-4] + '-probabilities.tif'
    write_fei_mock(prob, scale, output_fn)
    
def write_fei_mock(image, scale, fn):
    image2 = np.pad(image, ((0, 70), (0, 0)), mode='constant',
                    constant_values=0)
    scale_hdr = b'\r\nDate=13/10/2017\r\n[Scan]\r\nPixelHeight='
    scale_str = scale_hdr + str(scale).encode('ascii') + b'\r\n'
    iio.imsave(fn, image2)
    with open(fn, 'ab') as fout:
        fout.write(scale_str)
        
for original_fn, prob in zip(fns1 + fns2, probs):
    scale = get_scale(original_fn)
    output_fn = original_fn[:-4] + '-probabilities.tif'
    write_fei_mock(prob, scale, output_fn)
    
from glob import glob
fnsprob = glob('adam-mix4/*-probabilities.tif')
fnsprob[:5]
prob0 = iio.imread(fnsprob[0], format='fei')
prob0.meta
plt.imshow(prob0, cmap='magma')
from skan import pipe
get_ipython().run_line_magic('pinfo', 'pipe.process_images')
t0 = filters.threshold_otsu(prob0)
plt.imshow(t0)
t0.shape
t0 = filters.threshold_otsu(prob0) < prob0
plt.imshow(t0)
results = []
from tqdm import tqdm
for probfile in tqdm(fnsprob):
    image = iio.imread(probfile, format='fei')
    scale = image.meta['Scan']['PixelHeight']
    pixel_smoothing_radius = int(round(5e-9 / scale))  # 5nm
    image = filters.gaussian(image, sigma=pixel_smoothing_radius)
    thresholded = image > filters.threshold_otsu(image)
    skeleton = morphology.skeletonize(thresholded)
    quality = feature.shape_index(image, sigma=0, mode='reflect')
    skeleton = skeleton * quality
    framedata = csr.summarise(skeleton, spacing=scale)
    framedata['squiggle'] = np.log2(framedata['branch-distance'] /
                                    framedata['euclidean-distance'])
    framedata['scale'] = scale
    framedata.rename(columns={'mean pixel value': 'mean shape index'},
                     inplace=True)
    framedata['filename'] = probfile
    results.append(framedata)
                     
from skan import csr
for probfile in tqdm(fnsprob):
    image = iio.imread(probfile, format='fei')
    scale = image.meta['Scan']['PixelHeight']
    pixel_smoothing_radius = int(round(5e-9 / scale))  # 5nm
    image = filters.gaussian(image, sigma=pixel_smoothing_radius)
    thresholded = image > filters.threshold_otsu(image)
    skeleton = morphology.skeletonize(thresholded)
    quality = feature.shape_index(image, sigma=0, mode='reflect')
    skeleton = skeleton * quality
    framedata = csr.summarise(skeleton, spacing=scale)
    framedata['squiggle'] = np.log2(framedata['branch-distance'] /
                                    framedata['euclidean-distance'])
    framedata['scale'] = scale
    framedata.rename(columns={'mean pixel value': 'mean shape index'},
                     inplace=True)
    framedata['filename'] = probfile
    results.append(framedata)
                     
sum(map(len, results))
full = pd.concat(results)
full.to_hdf('prob-troph-data.hdf')
full.to_hdf('prob-troph-data.hdf', key='table')
import seaborn.apionly as sns
get_ipython().run_line_magic('pinfo', 'sns.boxplot')
get_ipython().run_line_magic('pwd', '')
full['mean shape index'].hist()
get_ipython().run_line_magic('pwd', '')
schizont_files = glob('adam-oli-schizont-30sec/*.tif')
len(schizont_files)
schizont_files = list(filter(lambda x: not x.endswith('01.tif'),
                             schizont_files))
                             
len(schizont_files)
prob0.shape
schizont_array = np.array([iio.imread(fn) for fn in schizont_files])
np.max(schizont_array)
from gala import imio
imio.write_h5_stack(schizont_array, 'adam-oli-schizont.h5',
                    compression='lzf', group='volume')
                    
import h5py
f = h5py.File('adam-oli-schizont_Probabilities.h5')
f.keys()
list(f.keys())
schizont_predictions = np.array(f['exported_data'])
schizont_predictions.shape
schizont_predictions = schizont_predictions.squeeze()
plt.imshow(schizont_predictions[0], cmap='magma')
plt.imshow(schizont_predictions[5], cmap='magma')
plt.imshow(schizont_predictions[-5], cmap='magma')
def write_fei_mock(image, scale, fn):
    image2 = np.pad(image, ((0, 70), (0, 0)), mode='constant',
                    constant_values=0)
    scale_hdr = b'\r\nDate=13/10/2017\r\n[Scan]\r\nPixelHeight='
    scale_str = scale_hdr + str(scale).encode('ascii') + b'\r\n'
    iio.imsave(fn, image2)
    with open(fn, 'ab') as fout:
        fout.write(scale_str)
        
for fn, image in zip(schizont_files, schizont_predictions):
    image = image[10:-10, 10:-80]
    fnout = fn[:-4] + '-predictions.tif'
    scale = get_scale(fn)
    write_fei_mock(image, scale, fnout)
    
fnsprob = glob('adam-oli-schizont-30sec/*-probabilities.tif')
len(fnsprob)
fnsprob = glob('adam-oli-schizont-30sec/*-predictions.tif')
len(fnsprob)
full.to_hdf('prob-troph-data-mix4.hdf', key='table')
results = []
for probfile in tqdm(fnsprob):
    image = iio.imread(probfile, format='fei')
    scale = image.meta['Scan']['PixelHeight']
    pixel_smoothing_radius = int(round(5e-9 / scale))  # 5nm
    image = filters.gaussian(image, sigma=pixel_smoothing_radius)
    thresholded = image > filters.threshold_otsu(image)
    skeleton = morphology.skeletonize(thresholded)
    quality = feature.shape_index(image, sigma=0, mode='reflect')
    skeleton = skeleton * quality
    framedata = csr.summarise(skeleton, spacing=scale)
    framedata['squiggle'] = np.log2(framedata['branch-distance'] /
                                    framedata['euclidean-distance'])
    framedata['scale'] = scale
    framedata.rename(columns={'mean pixel value': 'mean shape index'},
                     inplace=True)
    framedata['filename'] = probfile
    results.append(framedata)
                     
full2 = pd.concat(results)
full2.to_hdf('prob-troph-data.hdf', key='table')
full2.head()
def infection_status(filename):
    if 'Uninf' in filename:
        return 'normal'
    else:
        return 'infected'
    
set(full2['filename'].apply(infection_status))
full2_proc = pd.read_hdf('prob-troph-data-preprocess.hdf', key='pre0')
full2.shape
full2_proc.shape
set(full2_proc['infection'])
set(full2_proc['cell number'])
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', '~/projects/skan-scripts/')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('run', '-i malaria-skeleton-troph.py')
fig, axes = plt.subplots(2, 2)
ax = axes.ravel()
datar['filename'][0]
datar['filename'].loc[0]
str(datar['filename'].loc[0])
type(datar['filename'])
datar.columns
len(datar['filename'])
type(datar['filename'].loc[0])
len(datar['filename'].loc[0])
len(datar['filename'].iloc[0])
len(data['filename'].iloc[0])
list(set(datar['filename']))[0]
image = iio.imread('/Users/jni/Dropbox/data1/malaria/adam-oli-schizont-30sec/Shchizont4_UninfRBC7_06.tif', format='fei')
ax[0].imshow(image, cmap='gray')
ax[0].set_axis_off()
scalenm = image.meta['Scan']['PixelHeight'] * 10**9
scalenm
image.shape
image.shape[1] / 5
image.shape[1] / 5 / scalenm
ax[0].plot((1136, 1136 + image.shape[1] / 5 / scalenm), (960, 960),
           c='k', lw=2)
           
a = ax[0]
a.lines
a.lines.remove(0)
a.lines.pop()
fig.canvas.draw()
ax[0].clear()
ax[0].imshow(image, cmap='gray')
ax[0].set_axis_off()
500 / scalenm
ax[0].plot((1136, 1136 + 300 / scalenm), (960, 960),
           c='k', lw=2)
           
pixel_blur = int(round(0.1 * 50/scalenm))
pixel_blur
pixel_blur = 0.1 * 50/scalenm
pixel_blur
image_blurred = ndi.gaussian_filter(image, sigma=pixel_blur)
from skan.vendored import thresholding
image_thresholded = thresholding.threshold_sauvola(image_blurred, window_size=46, k=0.075)
image_thresholded = thresholding.threshold_sauvola(image_blurred, window_size=47, k=0.075)
image_thresholded = thresholding.threshold_sauvola(image_blurred, window_size=46*2 + 1, k=0.075)
skeleton = morphology.skeletonize(image_thresholded)
skeleton = morphology.skeletonize(image_thresholded.astype(np.uint8))
image_thresholded = thresholding.threshold_sauvola(image_blurred, window_size=46*2 + 1, k=0.075) < image_blurred
skeleton = morphology.skeletonize(image_thresholded)
viz = np.zeros(image.shape + (3,), dtype=float32)
viz = np.zeros(image.shape + (3,), dtype=float)
viz[image_thresholded] = [1, 1, 1]
viz[skeleton] = [1, 0, 0]
ax[1].imshow(viz)
ax[1].set_axis_off()
ax[1].plot((1136, 1136 + 300 / scalenm), (960, 960),
           c='k', lw=2)
           
ax[1].plot((1136, 1136 + 300 / scalenm), (960, 960),
           c='g', lw=2)
           
_, bins = np.histogram(datar['branch distance (nm)'], bins='auto')
for inf, df in datar.sort_values(by='infection', ascending=False).groupby('infection', sort=False):
    ax[2].hist(df['branch distance (nm)'], bins=bins, normed=True, alpha=0.5, label=inf)
    
ax.legend()
ax[2].legend()
ax[2].set_xlabel('branch distance (nm)')
ax[2].set_ylabel('density')
import seaborn.apionly as sns
sns.stripplot(x='infection', y='branch distance (nm)', data=cellmeans,
              jitter=True, order=('normal', 'infected'), ax=ax[3])
              
ax[3].set_xlabel('infection status')
ax[3].set_ylabel('mean branch distance by cell (nm)')
ax[3].set_ylabel('mean branch distance\nby cell (nm)')
fig.tight_layout()
fig.savefig('fig2.png', dpi=600)
fig.savefig('fig2.png', dpi=900)
int(np.ceil(50/scalenm))
r = int(np.ceil(50/scalenm))
ax[0].imshow(image, cmap='gray')
ax[0].imshow(image[20:-20, 20:-20], cmap='gray')
ax[0].plot((1096, 1096 + 300 / scalenm), (960, 960),
           c='k', lw=2)
           
from skan.pre import threshold
s = r * 0.1
thresholded = threshold(image, sigma=s, radius=r, offset=0.075, smooth_method='gaussian')
skeleton = morphology.skeletonize(thresholded)
thresholded = threshold(image[20:-20, 20:-20], sigma=s, radius=r, offset=0.075, smooth_method='gaussian')
skeleton = morphology.skeletonize(thresholded)
plt.figure(); plt.imshow(skeleton)
ax[1].imshow(thresholded, cmap='gray')
skel = np.zeros(thresholded.shape + (3,), dtype=float)
skel[thresholded] = [1, 1, 1]
fat_skeleton = morphology.binary_dilation(skeleton)
skel[fat_skeleton] = [1, 0, 0]
ax[1].imshow(skel)
ax[1]._lines.pop()
ax[1].lines.pop()
ax[1].lines.pop()
ax[1].lines
ax[1].plot((1096, 1096 + 300 / scalenm), (960, 960),
           c='g', lw=2)
           
ax[1].set_xlim(0, thresholded.shape[1])
ax[1].set_ylim(thresholded.shape[0], 0)
ax[1].lines.pop()
ax[1].plot((1096, 1096 + 300 / scalenm), (940, 940),
           c='g', lw=2)
           
ax[0].lines.pop()
ax[0].lines
ax[0].lines.pop()
ax[0].set_xlim(0, thresholded.shape[1])
ax[0].set_ylim(thresholded.shape[0], 0)
ax[0].plot((1096, 1096 + 300 / scalenm), (940, 940),
           c='k', lw=2)
           
fig.savefig('fig2.png', dpi=600)
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', 'results')
image0 = io.imread('results/test-image0.png')
image0.shape
plt.imshow(image0)
plt.imshow(image0)
plt.figure()
plt.imshow(image0)
image0 = io.imread('/Users/jni/Dropbox/data1/skeletons/skeleton0.png')
plt.imshow(image0)
plt.show()
plt.imshow(image0)
fig, axes = plt.subplots(2, 3)
fig, axes = plt.subplots(3, 2)
ax = axes.ravel()
ax[0].imshow(image0, cmap='gray')
ax[0].set_axis_off()
ax[0].set_axis_on()
ax[0].set_xticks([])
ax[0].set_yticks([])
get_ipython().run_line_magic('pinfo', 'csr.skeleton_to_csgraph')
g, idx, deg = csr.skeleton_to_csgraph(image0)
import networkx as nx
gx = nx.from_scipy_sparse_matrix(g)
ax[1].imshow(deg, cmap='magma')
deg2 = np.clip(deg, 0, 3)
ax[1].imshow(deg2, cmap='magma')
from mpl_toolkits.axes_grid1 import make_axes_locatable
ax1 = make_axes_locatable(ax[1])
1/7
cax3 = ax1.append_axes("right", size="14%", pad=0.05)
cbar3 = plt.colorbar(_272, cax=cax3, ticks=np.arange(0, 4), bounds=np.arange(0.5, 3, 1))
get_ipython().run_line_magic('pinfo', 'plt.colorbar')
cbar3 = plt.colorbar(_272, cax=cax3, values=np.arange(0, 4), boundaries=np.arange(-0.5, 3.51, 1))
cbar3 = plt.colorbar(_272, cax=cax3, values=np.arange(0, 4), boundaries=np.arange(-0.5, 3.51, 1), ticks=np.arange(0, 4))
get_ipython().run_line_magic('pinfo', 'plt.colorbar')
get_ipython().run_line_magic('pinfo', 'cbar3.ticklocation')
cbar3.ticklocation
cbar3.set_ticklabels(['0', '1', '2', '>3'])
cbar3.set_ticklabels(['0', '1', '2', '3+'])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[0].set_title('bare skeleton')
ax[1].set_title('pixel neighbour count')
imfiji = io.imread('/Users/jni/Dropbox/data1/skeletons/Screen\ Shot\ 2017-07-06\ at\ 6.08.37\ pm.png')
imfiji = io.imread('/Users/jni/Dropbox/data1/skeletons/Screen Shot 2017-07-06 at 6.08.37 pm.png')
ax[1].imshow(imfiji)
ax[1].imshow(deg2, cmap='magma')
ax[2].imshow(imfiji)
ax[2].set_axis_off()
ax[2].set_title('Fiji screenshot')
fig, axes = plt.subplots(2, 3)
ax = axes.ravel()
ax[0].imshow(image0, cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(deg2, cmap='magma')
ax1 = make_axes_locatable(ax[1])
cax1 = ax1.append_axes("right", size="14%", pad=0.05)
cbar1 = plt.colorbar(_302, cax=cax1, values=np.arange(0, 4), boundaries=np.arange(-0.5, 3.51, 1), ticks=np.arange(0, 4))
ax[1].set_xticks([])
ax[1].set_yticks([])
cbar1.set_ticklabels(['0', '1', '2', '3+'])
ax[2].imshow(imfiji)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[3].imshow(image0); ax[3].set_xticks([]); ax[3].set_yticks([])
ax[3].imshow(image0, cmap='gray'); ax[3].set_xticks([]); ax[3].set_yticks([])
g
gx
list(gx.edges())
get_ipython().run_line_magic('pinfo', 'nx.draw')
get_ipython().run_line_magic('pinfo', 'nx.draw_networkx')
gx0 = nx.from_edgelist([(1, 2), (2, 3), (4, 5), (5, ' 3 '), ('  3  ', 9), (9, 10)])
get_ipython().run_line_magic('pinfo', 'nx.draw_networkx')
pos0 = {1: (3, 0), 2: (3, 1), 3: (3, 2), ' 3 ': (2, 3), '  3  ': (4, 3), 4: (0, 3), 5: (1, 3), 9: (5, 3), 10: (6, 3)}
nx.draw_networkx(gx0, pos=pos0, ax=ax[3])
ax[3].imshow(image0, cmap='gray'); ax[3].set_xticks([]); ax[3].set_yticks([])
ax[3].points
ax[3].__attrs__
ax3 = ax[3]
ax3.cla()
ax[3].imshow(image0, cmap='gray'); ax[3].set_xticks([]); ax[3].set_yticks([])
get_ipython().run_line_magic('pinfo', 'nx.draw_networkx')
get_ipython().run_line_magic('pinfo', 'nx.draw_networkx')
nx.draw_networkx(gx0, pos=pos0, ax=ax[3], node_size=100, font_size=8)
ax[4].imshow(image0, cmap='gray'); ax[4].set_xticks([]); ax[4].set_yticks([])
gx1 = nx.from_edgelist([(1, 2), (2, 3), (4, 5), (5, 3), (3, 9), (9, 10)])
pos1 = {1: (3, 0), 2: (3, 1), 3: np.mean([(3, 2), (2, 3), (4, 3), (3, 3)], axis=0), 4: (0, 3), 5: (1, 3), 9: (5, 3), 10: (6, 3)}
nx.draw_networkx(gx1, pos=pos1, ax=ax[4], node_size=100, font_size=8)
pos2 = {1: (3, 0), 2: (3, 1), 3: (2, 3), 4: (0, 3), 5: (1, 3), 9: (5, 3), 10: (6, 3)}
ax[5].imshow(image0, cmap='gray'); ax[5].set_xticks([]); ax[5].set_yticks([])
nx.draw_networkx(gx1, pos=pos2, ax=ax[5], node_size=100, font_size=8)
fig.tight_layout()
fig.savefig('sup-fig1.png', dpi=600)
