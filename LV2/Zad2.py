import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("C:/Users/brnar/Downloads/mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

mpg = data[:, 0]
cyl = data[:, 1]
disp = data[:, 2]
hp = data[:, 3]
drat = data[:, 4]
wt = data[:, 5]

plt.scatter(hp, mpg, s=wt*100, alpha=0.5)
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Gallon (mpg)')
plt.title('Potrošnja vs. Konjske snage (veličina točkice = težina)')
plt.show()

print("Minimalna potrošnja:", np.min(mpg))
print("Maksimalna potrošnja:", np.max(mpg))
print("Srednja potrošnja:", np.mean(mpg))

mpg_6cyl = mpg[cyl == 6]

print("Minimalna potrošnja (6 cilindara):", np.min(mpg_6cyl))
print("Maksimalna potrošnja (6 cilindara):", np.max(mpg_6cyl))
print("Srednja potrošnja (6 cilindara):", np.mean(mpg_6cyl))
