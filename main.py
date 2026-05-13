import difflib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from parlay import (
    generate_all_outcome_parlays,
    parlay_probability
)

df = pd.read_csv("UFC.csv")

# Create fighter ages
df["r_age"] = (
    pd.to_datetime(df["date"]) - pd.to_datetime(df["r_dob"])
).dt.days / 365

df["b_age"] = (
    pd.to_datetime(df["date"]) - pd.to_datetime(df["b_dob"])
).dt.days / 365

# Red corner win target
df["red_won"] = (df["winner"] == df["r_name"]).astype(int)

# Features used by model
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

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model trained!")


# Suggest closest fighter name
def find_closest_fighter_name(name):
    name = name.strip()

    all_names = pd.concat([
        df["r_name"],
        df["b_name"]
    ]).dropna().unique()

    exact_matches = [
        fighter for fighter in all_names
        if fighter == name
    ]

    if exact_matches:
        return exact_matches[0]

    close_matches = difflib.get_close_matches(
        name,
        all_names,
        n=1,
        cutoff=0.6
    )

    if close_matches:
        suggestion = close_matches[0]

        answer = input(
            f'Did you mean "{suggestion}"? y/n: '
        )

        if answer.lower() == "y":
            return suggestion

    raise ValueError(
        f"Fighter not found in dataset: {name}"
    )


# Get average fighter stats
def get_fighter_stats(name):
    name = find_closest_fighter_name(name)

    red_fights = df[df["r_name"] == name][[
        "r_str_acc",
        "r_td_avg",
        "r_height",
        "r_reach",
        "r_age"
    ]].rename(columns={
        "r_str_acc": "str_acc",
        "r_td_avg": "td_avg",
        "r_height": "height",
        "r_reach": "reach",
        "r_age": "age"
    })

    blue_fights = df[df["b_name"] == name][[
        "b_str_acc",
        "b_td_avg",
        "b_height",
        "b_reach",
        "b_age"
    ]].rename(columns={
        "b_str_acc": "str_acc",
        "b_td_avg": "td_avg",
        "b_height": "height",
        "b_reach": "reach",
        "b_age": "age"
    })

    fighter = pd.concat([
        red_fights,
        blue_fights
    ])

    stats = {
        "str_acc": fighter["str_acc"].mean(),
        "td_avg": fighter["td_avg"].mean(),
        "height": fighter["height"].mean(),
        "reach": fighter["reach"].mean(),
        "age": fighter["age"].mean()
    }

    if any(
        pd.isna(value)
        for value in stats.values()
    ):
        raise ValueError(
            f"Missing stats for fighter: {name}"
        )

    return stats


# Predict fight probabilities
def predict_fight(fighter_one, fighter_two):
    fighter_one = fighter_one.strip()
    fighter_two = fighter_two.strip()

    fighter_one_stats = get_fighter_stats(
        fighter_one
    )

    fighter_two_stats = get_fighter_stats(
        fighter_two
    )

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

    probability = model.predict_proba(fight)

    return probability[0][1], probability[0][0]


choice = input(
    "Choose mode: 1 = Single fight, 2 = Parlays: "
)

# Single fight mode
if choice == "1":

    fighter_one = input(
        "Pick first fighter: "
    )

    fighter_two = input(
        "Pick second fighter: "
    )

    fighter_one_prob, fighter_two_prob = (
        predict_fight(
            fighter_one,
            fighter_two
        )
    )

    print(
        f"{fighter_one}: "
        f"{fighter_one_prob * 100:.2f}%"
    )

    print(
        f"{fighter_two}: "
        f"{fighter_two_prob * 100:.2f}%"
    )

# Parlay mode
elif choice == "2":

    number_of_fights = int(
        input(
            "How many fights are you picking from? "
        )
    )

    parlay_size = int(
        input(
            "How many legs per parlay? "
        )
    )

    fights = []

    for i in range(number_of_fights):

        print(f"\nFight {i + 1}")

        fighter_one = input(
            "Pick first fighter: "
        )

        fighter_two = input(
            "Pick second fighter: "
        )

        fighter_one_prob, fighter_two_prob = (
            predict_fight(
                fighter_one,
                fighter_two
            )
        )

        print(
            f"{fighter_one}: "
            f"{fighter_one_prob * 100:.2f}%"
        )

        print(
            f"{fighter_two}: "
            f"{fighter_two_prob * 100:.2f}%"
        )

        # Store both outcomes
        fights.append([
            (fighter_one, fighter_one_prob),
            (fighter_two, fighter_two_prob)
        ])

    # Generate all parlays
    parlays = generate_all_outcome_parlays(
        fights,
        parlay_size
    )

    print("\nBest Parlays:\n")

    # Show top parlays
    for parlay in parlays[:20]:

        names = [
            fighter
            for fighter, probability in parlay
        ]

        combined_probability = (
            parlay_probability(parlay)
        )

        print(" + ".join(names))

        print(
            f"Combined probability: "
            f"{combined_probability * 100:.2f}%\n"
        )

else:
    print("Invalid choice.")