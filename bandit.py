import numpy
from methods import *
import matplotlib.pyplot as plt


def generate_reward(arm):
    generator, params = ARMS_GENS[arm]
    return generator(*params.values())


def get_mean(dist, params):
    if dist == numpy.random.beta:
        return params['alpha'] / (params['alpha'] + params['beta'])
    elif dist == numpy.random.binomial:
        return params['p'] * params['n']
    elif dist == numpy.random.normal:
        return params['loc']
    else:
        raise NotImplementedError


numpy.random.seed(42)

REWARD_TYPE = 'bernoulli'
if REWARD_TYPE == 'bernoulli':
    ARMS_GENS = [
        [numpy.random.binomial, dict(n=1, p=0.3)],
        [numpy.random.binomial, dict(n=1, p=0.5)],
        [numpy.random.binomial, dict(n=1, p=0.7)],
    ]
    ARMS_CNT = len(ARMS_GENS)
    methods = {
        'TS': BernThompsonSampling(ARMS_CNT),
        'Greedy': BernEpsilonGreedy(ARMS_CNT, 0),
        'EpsG (0.1)': BernEpsilonGreedy(ARMS_CNT, 0.1),
        'EpsG (0.3)': BernEpsilonGreedy(ARMS_CNT, 0.3),
        'EpsG (0.5)': BernEpsilonGreedy(ARMS_CNT, 0.5),
        'EpsG (0.7)': BernEpsilonGreedy(ARMS_CNT, 0.7),
    }
elif REWARD_TYPE == 'gaussian':
    ARMS_GENS = [
        [numpy.random.normal, dict(loc=5, scale=4)],
        [numpy.random.normal, dict(loc=6, scale=2)],
        [numpy.random.normal, dict(loc=2, scale=50)],
    ]
    ARMS_CNT = len(ARMS_GENS)
    methods = {
        'TS (ρ=0.1)': GausThompsonSampling(ARMS_CNT, rho=0.1),
        'TS (ρ=0.3)': GausThompsonSampling(ARMS_CNT, rho=0.3),
        'TS (ρ=0.5)': GausThompsonSampling(ARMS_CNT, rho=0.5),
        'TS (ρ=0.7)': GausThompsonSampling(ARMS_CNT, rho=0.7),
        'UCB': GausUCB(ARMS_CNT),
        'Greedy': BernEpsilonGreedy(ARMS_CNT, 0),
    }
else:
    raise NotImplementedError

ARMS_MEAN = np.array([
    get_mean(gen, param)
    for gen, param in ARMS_GENS
])
PERIODS = 100

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(11, 5)
fig.suptitle(f'{REWARD_TYPE.upper()} reward')

total_rewards = {name: [0] for name in methods}
regrets = {name: [0] for name in methods}
best_arm_prob = max(ARMS_MEAN)

for t in range(1, PERIODS + 1):
    for method_name, method in methods.items():
        arm = method.select_arm()
        reward = generate_reward(arm)
        method.update(arm, reward)

        total_rewards[method_name].append(total_rewards[method_name][-1] + reward)
        regrets[method_name].append(regrets[method_name][-1] + best_arm_prob - ARMS_MEAN[arm])

x = np.arange(0, PERIODS + 1)
for method_name, method in methods.items():
    ax1.plot(x, total_rewards[method_name], label=method_name)
for method_name, method in methods.items():
    ax2.plot(x[1:], regrets[method_name][1:], label=method_name)

ax1.set(title=f'Cumulative rewards', xlabel='period', ylabel='total reward')
ax2.set(title=f'Cumulative regrets', xlabel='period', ylabel='total regret')
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
plt.savefig(f'figs/{REWARD_TYPE}.png')
fig.show()
