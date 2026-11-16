"""
A/B testing with multi-armed bandits.

Four advertisement options (bandits) give Gaussian rewards with true means
``Bandit_Reward = [1, 2, 3, 4]``. Two algorithms are compared over
``NumberOfTrials = 20000`` trials:

* Epsilon-Greedy with epsilon decaying as 1/t
* Thompson Sampling for Gaussian rewards with known precision

All output goes through loguru instead of ``print()``.
"""
############################### LOGGER
import os
from abc import ABC, abstractmethod

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
EPSILON = 1.0          # starting epsilon, decayed as EPSILON / t
PRECISION = 1.0        # known precision (tau) of the reward noise, so sigma = 1
SEED = 42
OUTPUT_DIR = "output"


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Plots for the learning process and for comparing the two algorithms."""

    def __init__(self, output_dir=OUTPUT_DIR):
        """
        :param output_dir: folder where the figures are saved
        :type output_dir: str
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot1(self, algorithm):
        """
        Show the learning process of one algorithm.

        Left panel: running average reward on a linear scale.
        Right panel: the same on a log x-axis, so the early trials are visible.
        A dashed line marks the best possible mean reward.

        :param algorithm: a finished EpsilonGreedy or ThompsonSampling run
        :type algorithm: Bandit
        """
        trials = np.arange(1, len(algorithm.rewards) + 1)
        running_avg = np.cumsum(algorithm.rewards) / trials
        best = max(algorithm.p)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, scale in zip(axes, ["linear", "log"]):
            ax.plot(trials, running_avg, label="Running average reward")
            ax.axhline(best, color="grey", linestyle="--", label=f"Best mean = {best}")
            ax.set_xscale(scale)
            ax.set_xlabel("Trial" + (" (log scale)" if scale == "log" else ""))
            ax.set_ylabel("Average reward")
            ax.set_title(f"{algorithm.name}: {scale} scale")
            ax.legend()
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"plot1_{algorithm.name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info(f"Saved learning curve for {algorithm.name} to {path}")

        # how often each bandit was chosen over time
        fig, ax = plt.subplots(figsize=(8, 4.5))
        chosen = np.array(algorithm.chosen_bandits)
        for i in range(len(algorithm.p)):
            share = np.cumsum(chosen == i) / trials
            ax.plot(trials, share, label=f"Bandit {i} (mean {algorithm.p[i]})")
        ax.set_xscale("log")
        ax.set_xlabel("Trial (log scale)")
        ax.set_ylabel("Share of pulls so far")
        ax.set_title(f"{algorithm.name}: which bandit gets pulled")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"plot1_{algorithm.name}_choices.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info(f"Saved bandit choice plot for {algorithm.name} to {path}")

    def plot2(self, eg, ts):
        """
        Compare cumulative rewards and cumulative regrets of the two algorithms.

        :param eg: finished Epsilon-Greedy run
        :type eg: EpsilonGreedy
        :param ts: finished Thompson Sampling run
        :type ts: ThompsonSampling
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for algo in (eg, ts):
            axes[0].plot(np.cumsum(algo.rewards), label=algo.name)
            axes[1].plot(np.cumsum(algo.regrets), label=algo.name)
        axes[0].set_title("Cumulative reward")
        axes[0].set_ylabel("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        axes[1].set_ylabel("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("Trial")
            ax.legend()
        fig.tight_layout()
        path = os.path.join(self.output_dir, "plot2_comparison.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info(f"Saved comparison plot to {path}")

#--------------------------------------#


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy over several Gaussian bandits, with epsilon = EPSILON / t.

    With probability epsilon a random bandit is explored, otherwise the bandit
    with the highest estimated mean is chosen. Estimates are sample means.
    """

    name = "EpsilonGreedy"

    def __init__(self, p, epsilon=EPSILON, precision=PRECISION,
                 n_trials=NUMBER_OF_TRIALS, seed=SEED):
        """
        :param p: true mean reward of each bandit
        :type p: list
        :param epsilon: starting epsilon, divided by t at trial t
        :type epsilon: float
        :param precision: precision (1 / variance) of the reward noise
        :type precision: float
        :param n_trials: number of trials in the experiment
        :type n_trials: int
        :param seed: random seed for reproducibility
        :type seed: int
        """
        self.p = list(p)
        self.epsilon = epsilon
        self.precision = precision
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)

        self.k = len(self.p)
        self.estimates = np.zeros(self.k)   # estimated mean of each bandit
        self.counts = np.zeros(self.k)      # how many times each was pulled

        self.chosen_bandits = []
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        """
        :return: short description of the algorithm and its settings
        :rtype: str
        """
        return (f"EpsilonGreedy(bandits={self.p}, epsilon0={self.epsilon}, "
                f"decay=1/t, trials={self.n_trials})")

    def pull(self, bandit):
        """
        Draw a reward from the chosen bandit: N(mean, 1 / precision).

        :param bandit: index of the bandit
        :type bandit: int
        :return: observed reward
        :rtype: float
        """
        return self.rng.normal(self.p[bandit], 1 / np.sqrt(self.precision))

    def update(self, bandit, reward):
        """
        Update the sample mean of the chosen bandit (incremental formula).

        :param bandit: index of the bandit
        :type bandit: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[bandit] += 1
        n = self.counts[bandit]
        self.estimates[bandit] += (reward - self.estimates[bandit]) / n

    def choose(self, t):
        """
        Pick a bandit for trial t.

        :param t: current trial, starting at 1
        :type t: int
        :return: index of the chosen bandit
        :rtype: int
        """
        eps = self.epsilon / t
        if self.rng.random() < eps:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.estimates))

    def experiment(self):
        """
        Run all trials and store the chosen bandits, rewards and regrets.

        Regret at a trial is the best true mean minus the true mean of the
        chosen bandit.

        :return: self, so calls can be chained
        :rtype: EpsilonGreedy
        """
        logger.info(f"Starting {self!r}")
        best = max(self.p)
        for t in range(1, self.n_trials + 1):
            bandit = self.choose(t)
            reward = self.pull(bandit)
            self.update(bandit, reward)

            self.chosen_bandits.append(bandit)
            self.rewards.append(reward)
            self.regrets.append(best - self.p[bandit])
        logger.debug(f"{self.name} final estimates: {np.round(self.estimates, 3)}")
        logger.debug(f"{self.name} pulls per bandit: {self.counts.astype(int)}")
        return self

    def to_frame(self):
        """
        :return: one row per trial with columns Bandit, Reward, Algorithm
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame({
            "Bandit": self.chosen_bandits,
            "Reward": self.rewards,
            "Algorithm": self.name,
        })

    def report(self, output_dir=OUTPUT_DIR):
        """
        Save the rewards to CSV and log cumulative and average reward and regret.

        :param output_dir: folder for the CSV file
        :type output_dir: str
        :return: per-trial data
        :rtype: pandas.DataFrame
        """
        os.makedirs(output_dir, exist_ok=True)
        df = self.to_frame()
        path = os.path.join(output_dir, f"{self.name}_rewards.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")

        n = len(self.rewards)
        cum_reward = float(np.sum(self.rewards))
        cum_regret = float(np.sum(self.regrets))
        logger.info(f"[{self.name}] Cumulative reward: {cum_reward:,.2f}")
        logger.info(f"[{self.name}] Cumulative regret: {cum_regret:,.2f}")
        logger.info(f"[{self.name}] Average reward per trial: {cum_reward / n:.4f}")
        logger.info(f"[{self.name}] Average regret per trial: {cum_regret / n:.4f}")
        if cum_regret / n > 0.1:
            logger.warning(f"[{self.name}] average regret is high, the algorithm "
                           f"may not have settled on the best bandit")
        return df

#--------------------------------------#


class ThompsonSampling(Bandit):
    """
    Thompson Sampling for Gaussian rewards with known precision.

    Each bandit's mean has a Normal prior N(m0, 1 / lambda0). Because the
    reward precision tau is known, the posterior stays Normal:

        lambda_n = lambda0 + n * tau
        m_n      = (lambda0 * m0 + tau * sum(x)) / lambda_n

    At every trial one value is sampled from each posterior and the bandit
    with the largest sample is pulled.
    """

    name = "ThompsonSampling"

    def __init__(self, p, precision=PRECISION, prior_mean=0.0, prior_precision=1.0,
                 n_trials=NUMBER_OF_TRIALS, seed=SEED):
        """
        :param p: true mean reward of each bandit
        :type p: list
        :param precision: known precision tau of the reward noise
        :type precision: float
        :param prior_mean: prior mean m0 for every bandit
        :type prior_mean: float
        :param prior_precision: prior precision lambda0 for every bandit
        :type prior_precision: float
        :param n_trials: number of trials in the experiment
        :type n_trials: int
        :param seed: random seed for reproducibility
        :type seed: int
        """
        self.p = list(p)
        self.precision = precision
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)

        self.k = len(self.p)
        self.post_mean = np.full(self.k, prior_mean, dtype=float)
        self.post_precision = np.full(self.k, prior_precision, dtype=float)
        self.sum_x = np.zeros(self.k)
        self.counts = np.zeros(self.k)

        self.chosen_bandits = []
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        """
        :return: short description of the algorithm and its settings
        :rtype: str
        """
        return (f"ThompsonSampling(bandits={self.p}, tau={self.precision}, "
                f"prior=N({self.prior_mean}, 1/{self.prior_precision}), "
                f"trials={self.n_trials})")

    def pull(self, bandit):
        """
        Draw a reward from the chosen bandit: N(mean, 1 / precision).

        :param bandit: index of the bandit
        :type bandit: int
        :return: observed reward
        :rtype: float
        """
        return self.rng.normal(self.p[bandit], 1 / np.sqrt(self.precision))

    def sample(self):
        """
        Draw one value from each bandit's posterior.

        :return: one sample per bandit
        :rtype: numpy.ndarray
        """
        return self.rng.normal(self.post_mean, 1 / np.sqrt(self.post_precision))

    def update(self, bandit, reward):
        """
        Conjugate Normal update of the chosen bandit's posterior.

        :param bandit: index of the bandit
        :type bandit: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[bandit] += 1
        self.sum_x[bandit] += reward
        self.post_precision[bandit] = self.prior_precision + self.counts[bandit] * self.precision
        self.post_mean[bandit] = (
            self.prior_precision * self.prior_mean + self.precision * self.sum_x[bandit]
        ) / self.post_precision[bandit]

    def experiment(self):
        """
        Run all trials and store the chosen bandits, rewards and regrets.

        :return: self, so calls can be chained
        :rtype: ThompsonSampling
        """
        logger.info(f"Starting {self!r}")
        best = max(self.p)
        for _ in range(self.n_trials):
            bandit = int(np.argmax(self.sample()))
            reward = self.pull(bandit)
            self.update(bandit, reward)

            self.chosen_bandits.append(bandit)
            self.rewards.append(reward)
            self.regrets.append(best - self.p[bandit])
        logger.debug(f"{self.name} posterior means: {np.round(self.post_mean, 3)}")
        logger.debug(f"{self.name} pulls per bandit: {self.counts.astype(int)}")
        return self

    def to_frame(self):
        """
        :return: one row per trial with columns Bandit, Reward, Algorithm
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame({
            "Bandit": self.chosen_bandits,
            "Reward": self.rewards,
            "Algorithm": self.name,
        })

    def report(self, output_dir=OUTPUT_DIR):
        """
        Save the rewards to CSV and log cumulative and average reward and regret.

        :param output_dir: folder for the CSV file
        :type output_dir: str
        :return: per-trial data
        :rtype: pandas.DataFrame
        """
        os.makedirs(output_dir, exist_ok=True)
        df = self.to_frame()
        path = os.path.join(output_dir, f"{self.name}_rewards.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")

        n = len(self.rewards)
        cum_reward = float(np.sum(self.rewards))
        cum_regret = float(np.sum(self.regrets))
        logger.info(f"[{self.name}] Cumulative reward: {cum_reward:,.2f}")
        logger.info(f"[{self.name}] Cumulative regret: {cum_regret:,.2f}")
        logger.info(f"[{self.name}] Average reward per trial: {cum_reward / n:.4f}")
        logger.info(f"[{self.name}] Average regret per trial: {cum_regret / n:.4f}")
        if cum_regret / n > 0.1:
            logger.warning(f"[{self.name}] average regret is high, the algorithm "
                           f"may not have settled on the best bandit")
        return df




def comparison(output_dir=OUTPUT_DIR):
    """
    Run both algorithms on the same bandits, plot them, save one combined CSV
    and log which one did better.

    :param output_dir: folder for figures and CSV files
    :type output_dir: str
    """
    eg = EpsilonGreedy(BANDIT_REWARD).experiment()
    ts = ThompsonSampling(BANDIT_REWARD).experiment()

    eg_df = eg.report(output_dir)
    ts_df = ts.report(output_dir)

    combined = pd.concat([eg_df, ts_df], ignore_index=True)
    path = os.path.join(output_dir, "bandit_rewards.csv")
    combined.to_csv(path, index=False)
    logger.info(f"Saved combined results ({len(combined)} rows) to {path}")

    viz = Visualization(output_dir)
    viz.plot1(eg)
    viz.plot1(ts)
    viz.plot2(eg, ts)

    eg_regret, ts_regret = sum(eg.regrets), sum(ts.regrets)
    winner = eg if eg_regret < ts_regret else ts
    logger.success(f"Lower cumulative regret: {winner.name} "
                   f"(EpsilonGreedy={eg_regret:.2f}, ThompsonSampling={ts_regret:.2f})")


if __name__=='__main__':
    comparison()
