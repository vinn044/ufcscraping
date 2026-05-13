from itertools import combinations


def generate_parlays(picks, parlay_size):
    return list(combinations(picks, parlay_size))


def parlay_probability(parlay):
    probability = 1

    for fighter, win_probability in parlay:
        probability *= win_probability

    return probability