from itertools import combinations, product


# Generate normal combinations
def generate_parlays(picks, parlay_size):
    return list(combinations(picks, parlay_size))


# Multiply probabilities together
def parlay_probability(parlay):
    probability = 1

    for fighter, win_probability in parlay:
        probability *= win_probability

    return probability


# Generate all possible outcome parlays
def generate_all_outcome_parlays(
    fights,
    parlay_size
):

    # Select fight groups
    fight_combos = combinations(
        fights,
        parlay_size
    )

    all_parlays = []

    # Generate all winner combinations
    for fight_combo in fight_combos:

        outcome_combos = product(
            *fight_combo
        )

        for parlay in outcome_combos:
            all_parlays.append(parlay)

    # Sort highest probability first
    all_parlays.sort(
        key=parlay_probability,
        reverse=True
    )

    return all_parlays