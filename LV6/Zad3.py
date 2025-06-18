from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_data(n_samples, flagc):
    if flagc == 1:
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=365)
    elif flagc == 2:
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=148)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        X, y = datasets.make_blobs(n_samples=n_samples, centers=4, 
                                  cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=148)
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=0.05)
    else:
        X, y = np.empty((0, 2)), np.empty((0,))
    return X, y

n_samples = 25
flagc = 5
X, y = generate_data(n_samples, flagc)

method = "ward"

linked = linkage(X, method=method)

plt.figure(figsize=(10, 7))
dendrogram(linked, labels=[f"Podatak {i+1}" for i in range(len(X))])
plt.ylabel('Udaljenost')
plt.title(f'Dendrogram - metoda: {method}')
plt.show()
