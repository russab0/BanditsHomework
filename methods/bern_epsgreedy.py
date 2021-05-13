import random
import numpy as np
from .base import BaseBandit


class BernEpsilonGreedy(BaseBandit):
    def __init__(self, arms_cnt: int, eps: float = 0.7):
        self.eps = eps
        self.values = np.zeros(arms_cnt)
        super().__init__(arms_cnt)

    def select_arm(self):
        if random.random() < 1 - self.eps:
            return self.values.argmax()
        return random.randrange(0, self.arms_cnt)

    def update(self, arm: int, reward: float):
        old_value = self.values[arm]
        old_count = self.counts[arm]
        new_value = (old_value * old_count + reward) / (old_count + 1)  # from the formula of average
        self.values[arm] = new_value
        self.counts[arm] += 1
