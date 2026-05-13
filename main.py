import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("UFC.csv")

df["r_age"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["r_dob"])).dt.days / 365
df["b_age"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["b_dob"])).dt.days / 365

df["red_won"] = (df["winner"] == df["r_name"]).astype(int)

features = [
    "r_str_acc",
    "b_str_acc",
    "r_td_avg",
    "b_td_avg",
    "r_height",
    "b_height",
    "r_reach",
    "b_reach",
    "r_age",
    "b_age"
]

data = df.dropna(subset=features + ["red_won"])

X = data[features]
y = data["red_won"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model trained!")


def get_fighter_stats(name):
    fighter = df[df["r_name"] == name]

    return {
        "str_acc": fighter["r_str_acc"].mean(),
        "td_avg": fighter["r_td_avg"].mean(),
        "height": fighter["r_height"].mean(),
        "reach": fighter["r_reach"].mean(),
        "age": fighter["r_age"].mean()
    }



fighter_one = (input("Pick first fighter: "))
fighter_two = (input("Pick second fighter: "))


fighter_one_stats = get_fighter_stats(fighter_one)
fighter_two_stats = get_fighter_stats(fighter_two)

fight = pd.DataFrame([{
    "r_str_acc": fighter_one_stats["str_acc"],
    "b_str_acc": fighter_two_stats["str_acc"],
    "r_td_avg": fighter_one_stats["td_avg"],
    "b_td_avg": fighter_two_stats["td_avg"],
    "r_height": fighter_one_stats["height"],
    "b_height": fighter_two_stats["height"],
    "r_reach": fighter_one_stats["reach"],
    "b_reach": fighter_two_stats["reach"],
    "r_age": fighter_one_stats["age"],
    "b_age": fighter_two_stats["age"]
}])

prediction = model.predict(fight)
probability = model.predict_proba(fight)




print(f"{fighter_one}: {probability[0][1] * 100:.2f}%")
print(f"{fighter_two}: {probability[0][0] * 100:.2f}%")