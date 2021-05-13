# Bandits Homework Report

## Intro
**Author**: Ruslan Sabirov

**Group**: BS17-DS-02

**Course**: [S21] Statistical techniques for Data Science

**VCS**: https://github.com/russab0/BanditsHomework

## Implementation

I have implemented the task in a several distinct Python scripts:
- **bandit.py** does all configurations (reward distributions, algorimts types), runs experiments and plots results;

- **methods** folder consist of all methods that solve Multi-Armed Bandit problem:
    - base.py implements base class with methods _\_\_init\_\_(arms_cnt)_, _select_arm()_, _update(arm, reward)_;
    - bern-epsgreedy.py presents $\epsilon$-greedy approach for Bernoulli reward with custom $\epsilon$-parameter (when $\epsilon$ is set to 0, method becomes simple Greedy);
    - bern-thompson.py demonstrates Thompson Sampling for Bernoulli reward;
    - gaus-ucb.py shows UCB method for Gaussian reward (Normal-Gamma conjugate prior);
    - gaus-thompson.py introduces Thomson Sampleing for Gaussian reward with Normal-Gamma conjugate prior (with adjustable risk tolerance $\rho$).


## Experiments and results

Below I present parameters of the conducted experiments

|         |Bernoulli           | Gaussian  |
| ------------- |-------------| -----|
| **methods**   | Bernoulli(p=0.3) <br> Bernoulli(p=0.5)<br>Bernoulli(p=0.7) | Normal($\mu=5, \sigma=4$)<br>Normal($\mu=6, \sigma=2$)<br>Normal($\mu=2, \sigma=50$) |
| **arms**      | ThomsonSampling<br> Greedy <br> $\epsilon$-greedy ($\epsilon=0.1$) <br> $\epsilon$-greedy ($\epsilon=0.3$)<br> $\epsilon$-greedy ($\epsilon=0.5$)<br> $\epsilon$-greedy ($\epsilon=0.7$)      |  UCB<br>Greedy<br> ThompsonSampling ($\rho=0.1$)<br> ThompsonSampling ($\rho=0.3$)<br>ThompsonSampling ($\rho=0.5$)<br>ThompsonSampling ($\rho=0.7$)|
| **# periods** | 100       |   100 |

Each of two experiments was done for 100 periods: this number allows to see future tendency and details at the beginning of the process as well.

All the experiments were done with fixed seed (42) for reproducibility. In order to reproduce the same results one should clone the project from VCS, set REWARD_TYPE to `bernoulli` or `gaussian` and run bandit.py.

I run two experiments and plotted two figures, for cumulative reward and cumulative regret:
- Reward is an value returned by arm after pulling it. Cumulative reward is calculated as $\operatorname{Cumreward}_{t}(\theta)=\operatorname{Cumreward}_{t-1}(\theta) + \operatorname{reward}_{t}(\theta)$.
- Per-period regret is $\operatorname{regret}_{t}(\theta)=\max _{k} \theta_{k}-\theta_{x_{t}}$. Cumulative regret is defines as $\operatorname{Cumregret}_{t}(\theta)=\operatorname{Cumregret}_{t-1}(\theta) + \operatorname{regret}_{t}(\theta)$.

There are two figures below: one for each experiment.

![](https://i.imgur.com/iyFNR7a.png)

![](https://i.imgur.com/8n5a9FE.png)

Some observations from the pictures:
- Greedy approach is lack of rewards and full of regrets, that is very predictable.
- For the Bernoulli reward EpsG (0.1) outperformed Thomson Sampling. That could be due to the specfic combination of arms and seed.
- For the Bernoulli TS ($\rho = 0.1$) excels the most. The reason for that could be the same. As $\rho$ is risk tolerance parameter, this means that at this specific experiment configuration, there was almost no need in risking (rxploitation won against exploration).


## Sources
1. A Tutorial on Thompson Sampling https://github.com/iosband/ts_tutorial
1. BanditsBook https://github.com/johnmyleswhite/BanditsBook
1. Thompson Sampling Algorithms for Mean-Variance Bandits
   https://arxiv.org/pdf/2002.00232.pdf