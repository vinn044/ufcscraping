# how i predict win probability

## currently taking into account striking accuracy, avg, takedowns, and age, with a logistic regression approach, since the only output is whether fighter one or two wins.

# CLI interface

## you can now interact with the cli inputting your two fighters rather than having to update your script each time.

# parlay option

## I'm quite interested in the round robin system, specifically a method where you place multiple smaller parlays at once from a single set of for example, 3-8 picks.  Essentially, you're breaking combinations into smaller groups, this way being able to win even if some, but not all, choices lose. This is why I added a parlay option when using the CLI

## For now like I mentioned earlier, I chose a logistic regression approach since its clear that the output is binary. I'm not sure if that'll change, as I need to learn more about what algorithm is suited better for the prediction of two fighters. As of now I see that with my parlay setting, the model simply says that the parlays with the highest probability of winning consists of just picking the higher probability fighter in their individual fights. This most likely is going to be a problem considering that highest probability is not always the best, betting-wise.