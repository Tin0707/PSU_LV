import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/brnar/Downloads/mtcars.csv")

plt.figure(figsize=(8, 5))
sns.barplot(x="cyl", y="mpg", data=df, estimator="mean", ci=None)
plt.title("Prosječna potrošnja prema broju cilindara")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="cyl", y="wt", data=df)
plt.title("Distribucija težine prema broju cilindara")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="am", y="mpg", data=df)
plt.title("Potrošnja prema tipu mjenjača (0 = automatski, 1 = ručni)")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="hp", y="qsec", hue="am", data=df)
plt.title("Ubrzanje vs. Snaga (po tipu mjenjača)")
plt.show()
