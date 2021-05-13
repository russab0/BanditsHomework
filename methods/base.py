import abc
import numpy as np


class BaseBandit(abc.ABC):
    def __init__(self, arms_cnt: int, *args, **kwargs):
        self.arms_cnt = arms_cnt
        self.counts = np.zeros(arms_cnt)

    def select_arm(self):
        raise NotImplementedError

    def update(self, arm, reward):
        raise NotImplementedError
