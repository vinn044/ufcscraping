import pandas as pd

df = pd.read_csv("fight_details.csv")

print(df.head())
print(df.columns)
print(df.shape)
print(df["division"])
print(df.describe())