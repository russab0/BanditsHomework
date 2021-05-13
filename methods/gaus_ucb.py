import numpy as np
from .base import BaseBandit


class GausUCB(BaseBandit):
    def __init__(self, arms_cnt: int):
        self.values = np.zeros(arms_cnt)
        super().__init__(arms_cnt)

    def select_arm(self):
        for arm in range(self.arms_cnt):
            if self.counts[arm] == 0:
                return arm

        ucb_values = np.zeros(self.arms_cnt)
        total_counts = self.counts.sum()
        for arm in range(self.arms_cnt):
            bonus = np.sqrt((2 * np.log(total_counts)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.argmax()

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
