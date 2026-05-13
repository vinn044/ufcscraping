import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("UFC.csv")

df["red_won"] = (df["winner"] == df["r_name"]).astype(int)

X = df[[
    "r_str_acc",
    "b_str_acc",
    "r_td_avg",
    "b_td_avg"
]]

y = df["red_won"]

model = LogisticRegression()

model.fit(X, y)

print("Model trained!")


def get_fighter_stats(name):
    fighter = df[df["r_name"] == name]

    return {
        "str_acc": fighter["r_str_acc"].mean(),
        "td_avg": fighter["r_td_avg"].mean()
    }


fighter_one = "Islam Makhachev"
fighter_two = "Charles Oliveira"

fighter_one_stats = get_fighter_stats(fighter_one)
fighter_two_stats = get_fighter_stats(fighter_two)

fight = [[
    fighter_one_stats["str_acc"],
    fighter_two_stats["str_acc"],
    fighter_one_stats["td_avg"],
    fighter_two_stats["td_avg"]
]]

prediction = model.predict(fight)

probability = model.predict_proba(fight)

print(f"{fighter_one}: {probability[0][1] * 100:.2f}%")
print(f"{fighter_two}: {probability[0][0] * 100:.2f}%")