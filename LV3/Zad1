import pandas as pd

df = pd.read_csv("C:/Users/brnar/Downloads/mtcars.csv")

print(df.sort_values(by="mpg").head(5)[["car", "mpg"]])
print(df[df["cyl"] == 8].sort_values(by="mpg").head(3)[["car", "mpg"]])
print(df[df["cyl"] == 6]["mpg"].mean())
df["wt_lbs"] = df["wt"] * 1000
print(df[(df["cyl"] == 4) & (df["wt_lbs"] >= 2000) & (df["wt_lbs"] <= 2200)]["mpg"].mean())
print(df[df["am"] == 1].shape[0])
print(df[df["am"] == 0].shape[0])
print(df[(df["am"] == 0) & (df["hp"] > 100)].shape[0])
df["wt_kg"] = df["wt_lbs"] * 0.45359237
print(df[["car", "wt_kg"]])
