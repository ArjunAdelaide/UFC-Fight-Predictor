# Learning Notes

## Step 1: Define The Prediction Target

The raw dataset puts the winner in `fighter_1`.

For modeling, we convert each fight into:

```text
fighter_a vs fighter_b -> fighter_a_wins
```

Sometimes `fighter_a` is the winner. Sometimes `fighter_a` is the loser. This
keeps the target from being all `1`s.

## Step 2: Avoid Data Leakage

Data leakage means the model gets information it would not know at prediction
time.

Bad input for future prediction:

- fight result method
- round and finish time
- knockdowns from the fight being predicted
- strikes from the fight being predicted
- takedowns from the fight being predicted
- control time from the fight being predicted

Good input:

- historical averages from previous fights
- physical measurements
- age at the fight date
- previous win rate
- previous finish rate
- previous striking/takedown/control rates
- days since last fight

## Step 3: Use A Date-Based Test

Random train/test splits are dangerous here because they mix older and newer
fights. We want to simulate predicting the future, so the baseline trains on
older fights and tests on later fights.

The first split is:

```text
train: before 2024-01-01
test:  2024-01-01 and later
```

## Step 4: Start Simple

The first model is logistic regression implemented with NumPy. This is not the
most powerful model, but it is easy to inspect and a good baseline.

Once the feature pipeline is solid, stronger models become much easier to add.

## Step 5: Improve One Thing At A Time

A useful model improvement was training-time matchup flipping. For each training
row, the project also creates the opposite orientation:

```text
fighter_a vs fighter_b -> fighter_a_wins
fighter_b vs fighter_a -> 1 - fighter_a_wins
```

This teaches the model that matchup order should be symmetric. The future test
set is not doubled, so evaluation still measures one prediction per historical
fight.

Not every plausible feature helps. A last-5-fights feature experiment slightly
hurt the held-out result, so it was not kept in the baseline.

Stronger model classes also need to earn their place. Random forests and
gradient boosting were tested, but the NumPy logistic regression baseline still
had better log loss and Brier score on the future-dated test split.

The next accepted feature improvement was method-adjusted Elo. The original Elo
feature treats every win the same. The new rating moves a fighter's Elo a little
more for KO/TKO and submission wins than for decisions.

This did not improve accuracy, which dipped slightly from about 63.3% to 63.0%.
But it improved log loss and Brier score slightly. That matters because the
model is trying to produce useful probabilities, not just winner picks.
