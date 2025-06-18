from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

face = mpimg.imread('C:/Users/brnar/Downloads/example_grayscale.png')
if len(face.shape) == 3:
    face = np.mean(face, axis=2)

X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=10, n_init=10)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

plt.figure(1)
plt.imshow(face, cmap='gray')
plt.axis('off')
plt.title('Originalna slika')

plt.figure(2)
plt.imshow(face_compressed, cmap='gray')
plt.axis('off')
plt.title('Kvantizirana slika (10 klastera)')

plt.show()
