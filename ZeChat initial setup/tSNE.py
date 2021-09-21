
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import math
from sklearn.metrics import pairwise_distances

t0 = time()


### set directory to load continuous wavelet transform spectrogram matrices from multiple files:
os.chdir(r'C:\Folder\Containing\Data')

c = 0
sample_size = 1000

for file in os.listdir('.'):
    if not file.endswith('.csv'):
        continue
    data = np.loadtxt(open(file), delimiter=",")
    
    if np.all(np.isfinite(data)) == 0:
        print("%s contains NaN or Inf values" % file)
        continue
    
    idx = np.random.randint(len(data), size=sample_size)
    data = data[idx,:]
    data = shuffle(data, random_state=0)

    if c == 0:
        cwtmatr = data
        c = 1
    else:
        cwtmatr = np.concatenate((cwtmatr, data), axis=0)
        
    print('loaded %s' % file)

print('data loaded')


### change directory to save results
os.chdir(r'C:\Directory\For\Saving\Results')


### randomly sample and shuffle data for tsne
sample_size = 3000
idx = np.random.randint(len(cwtmatr), size=sample_size)
cwtmatr = cwtmatr[idx,:]

cwtmatr = shuffle(cwtmatr, random_state=0)


print('randomly selected and shuffled data')


## define JS Distance (square root of JS Divergence, a metric)
def jsd(p, q):
    m = (p + q) / 2
    return ((scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2)**0.5


### precompute pairwise distances:
cwtmatr_precomputed = pairwise_distances(cwtmatr, metric=jsd)  

print('precomputed data')

t1 = time()
print(t1-t0)


### tsne with precomputed distances:
tsne = manifold.TSNE(n_components=2, n_iter=1000, learning_rate=200, perplexity=30, early_exaggeration=12, metric='precomputed', random_state=0) 

cwtmatr_tsne = tsne.fit_transform(cwtmatr_precomputed)


np.savetxt('tsne.csv', cwtmatr_tsne, delimiter=",")

print('tsne completed')

t2 = time()
print(t2-t1)


### export and visualize results

vis_x = cwtmatr_tsne[:, 0]
vis_y = cwtmatr_tsne[:, 1]

plt.figure(figsize=(10, 10))

plt.scatter(vis_x, vis_y, s=1)

plt.savefig('tsne.png', dpi=100)

plt.show()





