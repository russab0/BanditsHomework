import numpy as np
from .base import BaseBandit


class BernThompsonSampling(BaseBandit):
    def __init__(self, arms_cnt: int):
        self.alphas = np.ones(arms_cnt)  # pseudo-counts of success
        self.betas = np.ones(arms_cnt)  # pseudo-counts of failure
        super().__init__(arms_cnt)

    def select_arm(self):
        for arm in range(self.arms_cnt):
            if self.counts[arm] == 0:
                return arm
        exp_prob = np.array([np.random.beta(alpha, beta)
                             for alpha, beta in zip(self.alphas, self.betas)])
        return exp_prob.argmax()

    def update(self, arm, reward):
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward
        self.counts[arm] += 1
