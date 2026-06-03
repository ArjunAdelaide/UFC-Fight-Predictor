# Methodology

This project predicts UFC matchups as:

```text
fighter_a vs fighter_b -> probability that fighter_a wins
```

The main modeling challenge is not the logistic regression itself. The main
challenge is building examples that resemble a real prediction setting, where
only pre-fight information is available.

## Raw Data Structure

The cleaned historical CSV stores each fight with:

```text
fighter_1 = winner
fighter_2 = loser
```

That structure is useful for recording fight history, but dangerous for model
training. If the model trained directly on `fighter_1` and `fighter_2`, the
target would be trivial because `fighter_1` always wins.

To avoid that shortcut, each fight is converted into a neutral matchup:

```text
fighter_a vs fighter_b -> fighter_a_wins
```

For each fight, the row is randomly oriented:

- winner as `fighter_a`, loser as `fighter_b`, target `fighter_a_wins = 1`
- loser as `fighter_a`, winner as `fighter_b`, target `fighter_a_wins = 0`

This forces the model to learn matchup signals instead of column position.

## Leakage-Safe Feature Engineering

The raw CSV includes post-fight stats such as:

- method
- round and finish time
- knockdowns
- significant strikes
- takedowns
- control time

Those values are only known during or after the fight. They cannot be used as
direct inputs for predicting that same fight.

The feature builder prevents that leakage by processing fights in chronological
order:

```text
sort fights by date
for each fight:
  read both fighters' histories before the fight
  create the pre-fight feature row
  assign the target
  update fighter histories with the completed fight
```

The key rule is:

```text
features first, history update second
```

For example, if Fighter A lands 100 significant strikes in Fight 10, those
strikes may affect Fighter A's profile for Fight 11, but not the pre-fight row
for Fight 10.

## Fighter Profiles

Each fighter has a running historical profile. As fights are processed, the
profile tracks previous results and rates, including:

- wins, losses, and win percentage
- finish, KO/TKO, submission, and decision rates
- significant strikes landed and absorbed
- takedowns landed and absorbed
- knockdowns for and against
- control time for and against
- average fight duration
- streak and recent form
- days since last fight
- simple Elo rating

The model mostly receives differences:

```text
fighter_a value - fighter_b value
```

Examples include `elo_diff`, `win_pct_diff`, `reach_cm_diff`,
`recent_win_pct_diff`, and `sig_landed_per_fight_diff`.

## Train/Test Split

The project uses a date-based split:

```text
train: fights before 2024-01-01
test:  fights on or after 2024-01-01
```

A random split would mix older and newer fights, which can make evaluation look
better than a true future-prediction setting. A date-based split better matches
the intended use case: train on past fights, evaluate on later fights.

## Baseline Model

The current model is logistic regression implemented with NumPy. It is not the
most powerful possible model, but it is a strong first baseline because it is:

- transparent
- dependency-light
- fast enough for this dataset
- easy to debug with coefficients

The model outputs:

```text
P(fighter_a_wins)
```

## Probability Evaluation

Accuracy alone is not enough for a probability model. A model can be accurate
while still producing poorly calibrated probabilities.

The training script also writes:

- `outputs/calibration_buckets.csv`
- `outputs/confidence_buckets.csv`

Calibration buckets ask:

```text
When the model predicts 60-70%, does Fighter A win around 60-70% of the time?
```

Confidence buckets ask:

```text
When the model is more confident, is it more accurate?
```

These summaries help separate winner-picking performance from probability
quality.

## Current Limitations

This is still a baseline project. It does not include betting odds, rankings,
injury context, short-notice fight flags, weight misses, training camps, or
market information.

The current Elo system is also intentionally simple. Future versions could
adjust Elo by method, opponent quality, recency, margin, or weight class.

The model should be treated as a learning and sports analytics project, not as
betting advice.
