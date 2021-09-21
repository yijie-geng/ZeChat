
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import mark_boundaries
from time import time
from skimage.filters import sobel, laplace
from skimage.measure import label
from skimage.color import label2rgb

### load data
os.chdir(r'C:\Folder\Containing\Data')
kde_data = np.loadtxt(open("kde for watershed segmentation.csv", "rb"), delimiter=",")

elevation_map = laplace(kde_data, ksize=3) 

for_mask = np.zeros_like(kde_data)

for_mask[kde_data < 1] = 1 
for_mask[kde_data == 1] = 0 

local_maxi = peak_local_max(1-elevation_map, min_distance=20, threshold_abs=1.0001, exclude_border=True, indices=False, labels=for_mask) 

markers = ndi.label(local_maxi)[0]

segmentation = watershed(elevation_map, markers)

### change directory to save results
os.chdir(r'C:\Directory\For\Saving\Results')

np.savetxt("watershed.csv", segmentation, delimiter=",")


fig = plt.figure(figsize=(10, 10))
ax = fig.gca()

ax.imshow(mark_boundaries(np.rot90(kde_data), np.rot90(segmentation), color=(0, 0, 0)), cmap=plt.cm.gray, interpolation='nearest')  

plt.savefig('watershed.png', dpi=100)

plt.show()










