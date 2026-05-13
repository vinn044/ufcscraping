import difflib
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier

from prediction_tracker import (
    save_prediction,
    show_predictions,
    update_prediction_result,
    show_accuracy
)

df = pd.read_csv("UFC.csv")

# Create fighter ages
df["r_age"] = (
    pd.to_datetime(df["date"]) - pd.to_datetime(df["r_dob"])
).dt.days / 365

df["b_age"] = (
    pd.to_datetime(df["date"]) - pd.to_datetime(df["b_dob"])
).dt.days / 365

# Create matchup difference features
df["age_diff"] = df["r_age"] - df["b_age"]
df["reach_diff"] = df["r_reach"] - df["b_reach"]
df["height_diff"] = df["r_height"] - df["b_height"]

df["str_acc_diff"] = df["r_str_acc"] - df["b_str_acc"]
df["splm_diff"] = df["r_splm"] - df["b_splm"]
df["sapm_diff"] = df["r_sapm"] - df["b_sapm"]

df["td_avg_diff"] = df["r_td_avg"] - df["b_td_avg"]
df["td_def_diff"] = df["r_td_def"] - df["b_td_def"]

df["str_def_diff"] = df["r_str_def"] - df["b_str_def"]
df["sub_avg_diff"] = df["r_sub_avg"] - df["b_sub_avg"]

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
    "b_age",

    "r_splm",
    "b_splm",

    "r_sapm",
    "b_sapm",

    "r_str_def",
    "b_str_def",

    "r_td_def",
    "b_td_def",

    "r_sub_avg",
    "b_sub_avg",

    "age_diff",
    "reach_diff",
    "height_diff",

    "str_acc_diff",
    "splm_diff",
    "sapm_diff",

    "td_avg_diff",
    "td_def_diff",

    "str_def_diff",
    "sub_avg_diff"
]

data = df.dropna(subset=features + ["red_won"])

X = data[features]
y = data["red_won"]

# Train random forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Create SHAP explainer for local prediction explanations
explainer = shap.TreeExplainer(model)

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
        "r_age",
        "r_splm",
        "r_sapm",
        "r_str_def",
        "r_td_def",
        "r_sub_avg"
    ]].rename(columns={
        "r_str_acc": "str_acc",
        "r_td_avg": "td_avg",
        "r_height": "height",
        "r_reach": "reach",
        "r_age": "age",
        "r_splm": "splm",
        "r_sapm": "sapm",
        "r_str_def": "str_def",
        "r_td_def": "td_def",
        "r_sub_avg": "sub_avg"
    })

    blue_fights = df[df["b_name"] == name][[
        "b_str_acc",
        "b_td_avg",
        "b_height",
        "b_reach",
        "b_age",
        "b_splm",
        "b_sapm",
        "b_str_def",
        "b_td_def",
        "b_sub_avg"
    ]].rename(columns={
        "b_str_acc": "str_acc",
        "b_td_avg": "td_avg",
        "b_height": "height",
        "b_reach": "reach",
        "b_age": "age",
        "b_splm": "splm",
        "b_sapm": "sapm",
        "b_str_def": "str_def",
        "b_td_def": "td_def",
        "b_sub_avg": "sub_avg"
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
        "age": fighter["age"].mean(),
        "splm": fighter["splm"].mean(),
        "sapm": fighter["sapm"].mean(),
        "str_def": fighter["str_def"].mean(),
        "td_def": fighter["td_def"].mean(),
        "sub_avg": fighter["sub_avg"].mean()
    }

    if any(
        pd.isna(value)
        for value in stats.values()
    ):
        raise ValueError(
            f"Missing stats for fighter: {name}"
        )

    return stats


# Explain what pushed the prediction up or down
def explain_prediction(fight):
    shap_values = explainer.shap_values(fight)

    # Class 1 means red fighter / fighter one wins
    if isinstance(shap_values, list):
        fighter_one_shap = shap_values[1][0]
    else:
        fighter_one_shap = shap_values[0, :, 1]

    explanation = pd.DataFrame({
        "feature": fight.columns,
        "value": fight.iloc[0].values,
        "shap_value": fighter_one_shap
    })

    explanation["impact"] = explanation["shap_value"].abs()

    explanation = explanation.sort_values(
        by="impact",
        ascending=False
    )

    print("\nTop reasons for prediction:\n")

    for _, row in explanation.head(10).iterrows():
        direction = (
            "pushed toward fighter one"
            if row["shap_value"] > 0
            else "pushed away from fighter one"
        )

        print(
            f'{row["feature"]}: {row["value"]:.2f} '
            f'→ {direction}'
        )

    print()


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
        "b_age": fighter_two_stats["age"],

        "r_splm": fighter_one_stats["splm"],
        "b_splm": fighter_two_stats["splm"],

        "r_sapm": fighter_one_stats["sapm"],
        "b_sapm": fighter_two_stats["sapm"],

        "r_str_def": fighter_one_stats["str_def"],
        "b_str_def": fighter_two_stats["str_def"],

        "r_td_def": fighter_one_stats["td_def"],
        "b_td_def": fighter_two_stats["td_def"],

        "r_sub_avg": fighter_one_stats["sub_avg"],
        "b_sub_avg": fighter_two_stats["sub_avg"],

        "age_diff": (
            fighter_one_stats["age"]
            - fighter_two_stats["age"]
        ),

        "reach_diff": (
            fighter_one_stats["reach"]
            - fighter_two_stats["reach"]
        ),

        "height_diff": (
            fighter_one_stats["height"]
            - fighter_two_stats["height"]
        ),

        "str_acc_diff": (
            fighter_one_stats["str_acc"]
            - fighter_two_stats["str_acc"]
        ),

        "splm_diff": (
            fighter_one_stats["splm"]
            - fighter_two_stats["splm"]
        ),

        "sapm_diff": (
            fighter_one_stats["sapm"]
            - fighter_two_stats["sapm"]
        ),

        "td_avg_diff": (
            fighter_one_stats["td_avg"]
            - fighter_two_stats["td_avg"]
        ),

        "td_def_diff": (
            fighter_one_stats["td_def"]
            - fighter_two_stats["td_def"]
        ),

        "str_def_diff": (
            fighter_one_stats["str_def"]
            - fighter_two_stats["str_def"]
        ),

        "sub_avg_diff": (
            fighter_one_stats["sub_avg"]
            - fighter_two_stats["sub_avg"]
        )
    }])

    probability = model.predict_proba(fight)

    explain_prediction(fight)

    return probability[0][1], probability[0][0]


choice = input(
    "Choose mode: 1 = Fight Predictions, 2 = Tracker: "
)

# Fight prediction mode
if choice == "1":

    number_of_fights = int(
        input("How many fights do you want to predict? ")
    )

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

        save_answer = input(
            "Save this prediction? y/n: "
        )

        if save_answer.lower() == "y":

            fight_date = input(
                "Enter fight date, like 2026-05-13: "
            )

            save_prediction(
                fight_date,
                fighter_one,
                fighter_two,
                fighter_one_prob,
                fighter_two_prob
            )

# Tracker mode
elif choice == "2":

    tracker_choice = input(
        "Tracker: 1 = Show predictions, "
        "2 = Update result, "
        "3 = Show accuracy: "
    )

    if tracker_choice == "1":
        show_predictions()

    elif tracker_choice == "2":
        show_predictions()

        row_number = int(
            input("Which row number are you updating? ")
        )

        actual_winner = input(
            "Who actually won? "
        )

        update_prediction_result(
            row_number,
            actual_winner
        )

    elif tracker_choice == "3":
        show_accuracy()

    else:
        print("Invalid tracker choice.")

else:
    print("Invalid choice.")