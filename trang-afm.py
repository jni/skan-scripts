import numpy as np
import pandas as pd
# coding: utf-8

# We are attempting some preliminary measurements from AFM images of the spectrin skeleton of red blood cells, either uninfected or infected with a *P. falciparum* stage III gametocyte.
# 
# First, we start by setting some constants. We don't know the height scale of the images, so we guess — 0 to 1 in the image intensity space will map to 0nm to 5nm in height. The horizontal scales were found by opening the images in Fiji.

# In[1]:


scale_infected = 2.1466  # nm
scale_uninfected = 2.6419  # nm
hscale = 5  # nm


# Next, we read in the image files:

# In[2]:

import os
os.chdir(os.path.expanduser('~/Dropbox/data1/malaria/trang-afm/gathered'))
from skimage import io
images = io.imread_collection('*.tif')
print(images.files)


# The first GIII image has too many imaging artefacts, so we move straight on to the second infected cell:

# In[3]:


giii1 = images[1]


# (Aside: set up the plotting environment)

# In[4]:


import matplotlib.pyplot as plt
import os
# plt.style.use()

# The images come colormapped using the AFMHot colormap (or something similar to it). We need single values per pixel, so we convert it to a grayscale image:

# In[5]:


from skimage import color


# In[6]:


lumiii1 = color.rgb2gray(giii1)


# In[7]:


plt.imshow(lumiii1)
plt.axis('off');


# Although they are barely perceptible, there is some low-level noise in the height image, so we perform some mild smoothing:

# In[8]:


from skimage import filters


# In[9]:


σ = 2
glumiii1 = filters.gaussian(lumiii1, sigma=σ)


# In[10]:


from scipy import ndimage
from skimage import morphology
import numpy as np


# Now we threshold the gaussian image to obtain the shape of the spectrin mesh. We use an offset median filter, but this step is probably the most ripe for optimisation. For example, one could use a *shape index* to identify tubular structures in the image.

# In[11]:


def threshold_function(image, **kwargs):
    footprint = morphology.disk(radius=kwargs.get('radius', 15))
    filtered = filters.median(image, footprint) - kwargs.get('offset', 0)
    return morphology.closing(image * 255 > filtered, np.ones((5, 5)))

biniii1 = threshold_function(glumiii1, radius=31, offset=7)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(biniii1)
ax[0].axis('off')
ax[1].imshow(glumiii1, cmap='magma')
ax[1].axis('off')
plt.tight_layout()


# Next, we skeletonise the thresholded image, and display the skeleton on top of the original image.

# In[12]:


skeliii1 = morphology.skeletonize(biniii1)

fig, ax = plt.subplots()
skeliii1_viz = giii1
skeliii1_viz[skeliii1] = [255, 0, 0]
ax.imshow(skeliii1_viz)
ax.axis('off')
plt.savefig('gamiii-skeleton-1.png')


# Looks reasonable! Let's make some measurements of the skeleton with the [`skan`](https://pypi.python.org/pypi/skan) package:

# In[13]:


import skan


# When a skeleton analysis needs to take height into account, `skan` expects a floating point value at each position in the skeleton. We can obtain this by multiplying the skeleton by the (smoothed) original image, and then by the scale.

# In[14]:


hskeliii1 = skeliii1 * glumiii1 * hscale


# In[15]:


stats_giii = skan.summarise(hskeliii1, spacing=scale_infected)


# Here's the dataset:

# In[16]:


stats_giii.describe()


# We want to exclude branches that go out to "tips", which can be implausibly short, instead only measuring links between junctions. This is encoded in the "branch type" variable, with type 2 being junction-junction branches.

# In[18]:


stats_giii[stats_giii['branch-type'] == 2].describe()


# Finally, some sections are missing branch points, and so register as exceedingly long branches (see that 454nm branch!). We can exclude these by measuring the ratio of branch distance to euclidean (straight line) distance between the branch points, and remove branches where this is too high.

# In[19]:


stats_giii['squiggle'] = np.log2(stats_giii['branch-distance'] / stats_giii['euclidean-distance'])


# In[20]:


stats_giii[(stats_giii['branch-type'] == 2) & (stats_giii['squiggle'] < 0.3)].describe()


# # Uninfected cell
# 
# Let's repeat the analysis for an uninfected red blood cell:

# In[21]:


lumrbc = color.rgb2gray(images[2])
plt.imshow(lumrbc, cmap='magma')
plt.axis('off');


# In[22]:


glumrbc = filters.gaussian(lumrbc, sigma=σ)


# In[23]:


binrbc = threshold_function(glumrbc)


# In[24]:


fig, ax = plt.subplots(1, 2)
ax[0].imshow(binrbc)
ax[1].imshow(glumrbc, cmap='magma')


# In[25]:


skelrbc = morphology.skeletonize(binrbc)

fig, ax = plt.subplots()
skelrbc_viz = images[2]
skelrbc_viz[skelrbc] = [255, 0, 0]
ax.imshow(skelrbc_viz)
ax.axis('off')
plt.savefig('urbc-skeleton.png')


# In[26]:


hskelrbc = skelrbc * hscale * glumrbc

stats_rbc = skan.summarise(hskelrbc, spacing=scale_uninfected)
stats_rbc['squiggle'] = np.log2(stats_rbc['branch-distance'] / stats_rbc['euclidean-distance'])


# In[27]:


stats_rbc.describe()


# In[28]:


stats_rbc[stats_rbc['branch-type'] == 2].describe()


# In[29]:


stats_rbc[(stats_rbc['branch-type'] == 2) & (stats_rbc['squiggle'] < 0.3)].describe()


# To summarise, we appear to have an increase of 13-21% in the mean/median length of spectrin in a GIII-infected cell, compared to an uninfected control.

# In[30]:


stats_giii[(stats_giii['branch-type'] == 2) & (stats_giii['squiggle'] < 0.3)].describe()


# In[39]:


stats_giii['infection status'] = 'infected (stage III gametocyte)'
stats_rbc['infection status'] = 'uninfected'
stats = pd.concat((stats_rbc, stats_giii))
stats['branch distance (nm)'] = stats['branch-distance']
stats_valid = stats[(stats['branch-type'] == 2) & (stats['squiggle'] < 0.3)]


# In[40]:


import seaborn.apionly as sns

sns.violinplot(data=stats_valid,
               x='infection status', y='branch distance (nm)',
               cut=0)


# In[41]:


stats_valid.groupby('infection status').median()['branch distance (nm)']


# In[42]:


stats_valid.groupby('infection status').count()


# In[45]:


import string
from itertools import cycle

def label_axes(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`
    Parameters
    ----------
    fig : Figure
         Figure object to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.
    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.lowercase
        
    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (.9, .9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)


# In[65]:


skelrbc.shape


# In[93]:


def imshow(image, axes, label=None, scale=None, scale_bar=None,
           textcolor='k', barcolor='w'):
    extent = None
    if scale is not None:
        ys, xs = np.array(image.shape[:2]) * scale
        print('extent:', xs, ys)
        extent = [0, xs, ys, 0]
        if scale_bar is not None:
            y = 0.9 * ys
            x0 = xs - scale_bar * 1.5
            x1 = xs - scale_bar * 0.5
            axes.hlines(y, x0, x1, colors=barcolor)
            print('scale bar coords:', y, x0, x1)
    if label is not None:
        y = 0.1 * ys
        x = xs + 20
        print('label coords:', x, y)
        axes.annotate(label, xy=(x, y), color=textcolor,
                      xycoords='data')
    axes.imshow(image, extent=extent)
    axes.axis('off')


# In[98]:


from matplotlib import gridspec

fig = plt.figure(figsize=(8, 6), dpi=300)
gs = gridspec.GridSpec(8, 6)
ax_rbc = plt.subplot(gs[:4, :3])
ax_iii = plt.subplot(gs[:4, 3:], sharey=ax_rbc)
ax_plt = plt.subplot(gs[4:, :])

imshow(skeliii1_viz, axes=ax_iii,
       label='B', scale=scale_infected, scale_bar=200)
imshow(skelrbc_viz, axes=ax_rbc,
       label='A', scale=scale_uninfected, scale_bar=200)
sns.violinplot(data=stats_valid,
               x='infection status', y='branch distance (nm)',
               cut=0, ax=ax_plt)
ax_plt.annotate('C', xy=(0.9, 0.9), xycoords='axes fraction')
plt.tight_layout()
fig.savefig('Figure-Gametocytes-Trang-AFM.png', dpi=300)


# In[ ]:




