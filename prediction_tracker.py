import os
import pandas as pd


PREDICTIONS_FILE = "saved_predictions.csv"


# Create saved_predictions.csv if it does not exist yet
def create_predictions_file():
    if not os.path.exists(PREDICTIONS_FILE):
        df = pd.DataFrame(columns=[
            "date",
            "fighter_one",
            "fighter_two",
            "fighter_one_probability",
            "fighter_two_probability",
            "predicted_winner",
            "predicted_probability",
            "actual_winner",
            "was_correct"
        ])

        df.to_csv(PREDICTIONS_FILE, index=False)


# Save one fight prediction
def save_prediction(
    date,
    fighter_one,
    fighter_two,
    fighter_one_probability,
    fighter_two_probability
):
    create_predictions_file()

    if fighter_one_probability > fighter_two_probability:
        predicted_winner = fighter_one
        predicted_probability = fighter_one_probability
    else:
        predicted_winner = fighter_two
        predicted_probability = fighter_two_probability

    new_prediction = pd.DataFrame([{
        "date": date,
        "fighter_one": fighter_one,
        "fighter_two": fighter_two,
        "fighter_one_probability": fighter_one_probability,
        "fighter_two_probability": fighter_two_probability,
        "predicted_winner": predicted_winner,
        "predicted_probability": predicted_probability,
        "actual_winner": "",
        "was_correct": ""
    }])

    df = pd.read_csv(PREDICTIONS_FILE)

    df = pd.concat(
        [df, new_prediction],
        ignore_index=True
    )

    df.to_csv(PREDICTIONS_FILE, index=False)

    print("\nPrediction saved!")
    print(f"Predicted winner: {predicted_winner}")
    print(f"Confidence: {predicted_probability * 100:.2f}%\n")


# Show saved predictions
def show_predictions():
    create_predictions_file()

    df = pd.read_csv(PREDICTIONS_FILE)

    if df.empty:
        print("\nNo predictions saved yet.\n")
        return

    print("\nSaved Predictions:\n")
    print(df)


# Update a saved prediction after the fight happens
def update_prediction_result(row_number, actual_winner):
    create_predictions_file()

    df = pd.read_csv(PREDICTIONS_FILE)

    if df.empty:
        print("\nNo predictions found.\n")
        return

    if row_number < 0 or row_number >= len(df):
        print("\nInvalid row number.\n")
        return

    predicted_winner = df.loc[row_number, "predicted_winner"]

    df.loc[row_number, "actual_winner"] = actual_winner
    df.loc[row_number, "was_correct"] = (
        predicted_winner == actual_winner
    )

    df.to_csv(PREDICTIONS_FILE, index=False)

    print("\nPrediction updated!")
    print(f"Predicted winner: {predicted_winner}")
    print(f"Actual winner: {actual_winner}")
    print(f"Correct: {predicted_winner == actual_winner}\n")


# Show overall model accuracy from saved predictions
def show_accuracy():
    create_predictions_file()

    df = pd.read_csv(PREDICTIONS_FILE)

    completed = df[df["actual_winner"].notna()]
    completed = completed[completed["actual_winner"] != ""]

    if completed.empty:
        print("\nNo completed predictions yet.\n")
        return

    correct = completed[completed["was_correct"] == True]

    accuracy = len(correct) / len(completed)

    print("\nModel Accuracy:\n")
    print(f"Completed predictions: {len(completed)}")
    print(f"Correct predictions: {len(correct)}")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")