# Homework 2: A/B Testing with Bandits

Four ads (bandits) give Gaussian rewards with true means `[1, 2, 3, 4]` and variance 1. I compare Epsilon-Greedy and Thompson Sampling over 20,000 trials.

## How to run

```bash
pip install -r requirements.txt
python Bandit.py
```

Everything is saved to the `output/` folder:

- `bandit_rewards.csv` has all trials from both algorithms, with columns `Bandit, Reward, Algorithm`
- `EpsilonGreedy_rewards.csv` and `ThompsonSampling_rewards.csv` have each algorithm on its own
- `plot1_<Algorithm>.png` shows the running average reward on linear and log scales
- `plot1_<Algorithm>_choices.png` shows how often each bandit was picked over time
- `plot2_comparison.png` shows cumulative reward and cumulative regret for both algorithms

Cumulative reward, cumulative regret and the per-trial averages are logged with loguru.

## Choices

- Epsilon-Greedy: epsilon starts at 1 and decays as 1/t. Estimates are sample means.
- Thompson Sampling: known precision tau = 1 (the true noise level). Each bandit has a prior N(0, 1). The posterior after n pulls has precision `1 + n*tau` and mean `tau * sum(x) / (1 + n*tau)`.
- Regret at each trial is `4 - true mean of the chosen bandit`.
- Seed is 42 so the results repeat.

## Results (seed 42)

| Algorithm | Cumulative reward | Cumulative regret |
|---|---|---|
| Epsilon-Greedy | 73,907 | 6,181 |
| Thompson Sampling | 79,967 | 7 |

Thompson Sampling found the best ad almost at once. Epsilon-Greedy with 1/t decay got stuck on the mean-3 ad for about 6,000 trials. With 1/t, exploration dies out very fast: after 100 trials epsilon is already 0.01. So if the greedy choice is wrong early on, it takes a long time to fix it.

## Bonus: a better implementation plan

1. **Slower epsilon decay.** Use something like `epsilon = min(1, c * k / t)` with c around 5 to 10, or start with a short phase where every ad is pulled a few times. This stops Epsilon-Greedy from locking onto a wrong ad too early.
2. **Optimistic starting values.** Starting all estimates high (for example 10) makes the greedy step try every ad before it settles.
3. **Add UCB1 as a third baseline.** It explores based on uncertainty and has no epsilon to tune.
4. **Average over many runs.** One run with one seed can be lucky or unlucky. Running each algorithm 100 times and plotting the mean regret with a confidence band gives a fairer comparison.
5. **Unknown precision.** In a real A/B test we don't know the noise level. A Normal-Gamma prior lets Thompson Sampling learn the mean and the precision together.
6. **Less repeated code.** Both classes share `pull`, `to_frame` and `report`. These could live in a shared base class, so each algorithm only defines how it picks a bandit and how it updates.
