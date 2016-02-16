# bandit data

import numpy as np

def gen_static_uniform(k = 2, T = 10):
	temp = np.random.rand(k)
	rewards = np.zeros((T, k))
	for t in range(T):
		rewards[t] += temp
	return rewards

def gen_static_beta(k = 2, T = 10, a = 1, b = 10):
	temp = np.random.beta(a, b, k)
	rewards = np.zeros((T, k))
	for t in range(T):
		rewards[t] += temp
	return rewards
