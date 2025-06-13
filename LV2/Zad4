import numpy as np
import matplotlib.pyplot as plt

def kvadratna_sahovnica(velicina_kvadrata, broj_redaka, broj_stupaca):
    crni = np.zeros((velicina_kvadrata, velicina_kvadrata))
    bijeli = np.ones((velicina_kvadrata, velicina_kvadrata)) * 255
    redovi = []
    for i in range(broj_redaka):
        red = []
        for j in range(broj_stupaca):
            if (i + j) % 2 == 0:
                red.append(crni)
            else:
                red.append(bijeli)
        redovi.append(np.hstack(red))
    return np.vstack(redovi)

img = kvadratna_sahovnica(20, 8, 8)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
