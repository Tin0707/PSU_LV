import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import make_blobs

def generate_data(n_samples, flagc):
    if flagc == 1:
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=365)
    elif flagc == 2:
        X, y = make_blobs(n_samples=n_samples, random_state=148)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        X, y = make_blobs(n_samples=n_samples, centers=4,
                          cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=148)
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=0.05)
    else:
        X, y = np.empty((0, 2)), np.empty((0,))
    return X, y

X, y = generate_data(500, flagc=5)

inertia_values = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.plot(range(1, 21), inertia_values, marker='o')
plt.title('Elbow metoda - KMeans')
plt.xlabel('Broj klastera (k)')
plt.ylabel('Vrijednost kriterijske funkcije (inertia)')
plt.grid(True)
plt.show()
