import numpy as np
from .base import BaseBandit


class GausThompsonSampling(BaseBandit):
    def __init__(self, arms_cnt: int, rho: float = 0.5):
        self.rho = rho
        self.means = np.zeros(arms_cnt)  # predicted mean
        self.alphas = np.ones(arms_cnt) / 2  # shape parameter
        self.betas = np.ones(arms_cnt) / 2  # rate parameter
        super().__init__(arms_cnt)

    def select_arm(self):
        mini_count = self.counts.min()
        mini_count_id = self.counts.argmin()
        if mini_count == 0:
            self.counts[mini_count_id] = 1
            return mini_count_id

        tau = np.array([
            np.random.gamma(shape=alpha, scale=1 / beta)
            for alpha, beta in zip(self.alphas, self.betas)
        ])
        theta = np.array([
            np.random.normal(loc=mean, scale=1 / cnt)
            for mean, cnt in zip(self.means, self.counts)
        ])
        exp_prob = theta * self.rho - 1 / tau
        return exp_prob.argmax()

    def update(self, arm, reward):
        if self.counts[arm] == 1:
            self.means[arm] = reward

        prev_mean = self.means[arm]
        self.means[arm] = self.counts[arm] / (self.counts[arm] + 1) * prev_mean + reward / (self.counts[arm] + 1)
        self.alphas[arm] += 0.5
        self.betas[arm] += self.counts[arm] / (self.counts[arm] + 1) * (reward - prev_mean) ** 2 / 2
        self.counts[arm] += 1
