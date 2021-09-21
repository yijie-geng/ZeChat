
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from sklearn import manifold
import scipy.stats as st
from time import time

### load data from a folder containing multiple ktsne data files
os.chdir(r'C:\Folder\Containing\Data\Files')

c = 0
sample_size = 1000

for file in os.listdir('.'):
    if not file.endswith('.csv'):
        continue
    
    data = np.loadtxt(open(file), delimiter=",")

    idx = np.random.randint(len(data), size=sample_size)
    data = data[idx,:]

    if c == 0:
        vis_data = data
        c = 1
    else:
        vis_data = np.concatenate((vis_data, data), axis=0)
        
    print('load_' + str(file))

print('data loaded')


### change directory to save results
os.chdir(r'C:\Directory\For\Saving\Results')


## randomly sample ktSNE data to reduce computation
sample_size = 10000 
idx = np.random.randint(len(vis_data), size=sample_size)
vis_data = vis_data[idx,:]

print("data sampled")

t0= time()


vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]


## set min/max; leave enough spaces around the edge of the map for watershed segmentation
xmin, xmax = -140, 120 
ymin, ymax = -100, 100

xx, yy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j] 
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([vis_x, vis_y])
kernel = st.gaussian_kde(values)
kernel.set_bandwidth(bw_method = 0.1)

f = np.reshape(kernel(positions).T, xx.shape)
f_invert_1000 = 1 - f*1000

t1 = time()
print(t1-t0)

np.savetxt('kde for watershed segmentation.csv', f_invert_1000, delimiter=",")

plt.figure(figsize=(10, 10))
plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

cmap = colors.LinearSegmentedColormap.from_list("", ["white","royalblue","yellow","red"])

plt.imshow(np.rot90(f), cmap=cmap, extent=[xmin, xmax, ymin, ymax])
plt.contour(xx, yy, f, colors='k')
plt.colorbar()

plt.savefig('kde.png')

plt.show()











