import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")
gray = img[:, :, 0].copy()

brighter = np.clip(gray + 0.3, 0, 1)
rotated = np.rot90(gray, k=3)
mirrored = np.fliplr(gray)
factor = 10
small = gray[::factor, ::factor]
height, width = gray.shape
quarter = np.zeros_like(gray)
start = width // 4
end = width // 2
quarter[:, start:end] = gray[:, start:end]

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].imshow(gray, cmap="gray")
axs[0, 0].set_title("Original")

axs[0, 1].imshow(brighter, cmap="gray")
axs[0, 1].set_title("Posvijetljena")

axs[0, 2].imshow(rotated, cmap="gray")
axs[0, 2].set_title("Rotirana 90°")

axs[1, 0].imshow(mirrored, cmap="gray")
axs[1, 0].set_title("Zrcaljena")

axs[1, 1].imshow(small, cmap="gray")
axs[1, 1].set_title("Smanjena rezolucija")

axs[1, 2].imshow(quarter, cmap="gray")
axs[1, 2].set_title("Druga četvrtina")

for ax in axs.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()
