import pandas as pd

df = pd.read_csv("fight_details.csv")

print(df.head())
print(df.columns)
print(df.shape)

print(df[["r_name", "b_name", "method"]].head(20))
print(df.iloc[0])