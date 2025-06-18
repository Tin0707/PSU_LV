import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

def generate_data(n_samples, flagc):
    if flagc == 1:
        X, _ = datasets.make_blobs(n_samples=n_samples, random_state=365)
    elif flagc == 2:
        X, _ = datasets.make_blobs(n_samples=n_samples, random_state=148)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        X, _ = datasets.make_blobs(n_samples=n_samples, centers=4,
                                   cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=148)
    elif flagc == 4:
        X, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    elif flagc == 5:
        X, _ = datasets.make_moons(n_samples=n_samples, noise=0.05)
    else:
        X = np.empty((0, 2))
    return X

n_samples = 500
flagc = 5

X = generate_data(n_samples, flagc)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='purple', marker='X', s=200)
plt.title(f'KMeans klasteriranje - Generirani podaci (flagc={flagc})')
plt.xlabel('Značajka 1')
plt.ylabel('Značajka 2')
plt.grid(True)
plt.tight_layout()
plt.show()
