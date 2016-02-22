from math import sqrt
import numpy as np
from numpy.random import choice

class Bandit(object):
	def __init__(self, n_arms):
		self.n = n_arms
		self.arms = {}
		for i in range(self.n):
			self.arms[i] = {}
			self.arms[i]['observations'] = [1,0]

	def add_arm(self, arm_id = None):
		if not arm_id:
			self.n += 1
			arm_id = self.n
		else:
			arm_id = arm_id
		self.arms[arm_id] = {}
		self.arms[arm_id]['observations'] = [1,0]
		return arm_id

	def remove_arm(self, arm_id):
		assert arm_id in self.arms.keys()
		del self.arms[arm_id]

	def reset_arm(self, arm_id):
		self.arms[i]['observations'] = [1,0]

	def update(self, arm_id, reward):
		self.arms[arm_id]['observations'].append(reward)

	def get_expected_value(self, arm_id):
		return np.mean(self.arms[arm_id]['observations'])

	def get_variance(self, arm_id):
		return np.var(self.arms[arm_id]['observations'])

	def get_expected_values_all(self):
		return [self.get_expected_value(x) for x in  self.arms.keys()]

	def get_variances_all(self):
		return [self.get_variance(x) for x in self.arms.keys()]

class AB(Bandit):
	def __init__(self, n_arms, stop_after=30):
		super().__init__(n_arms)
		self.stop_after = stop_after
		
	def choose_arm(self):
		counts = [(len(self.arms[arm_id]['observations']), arm_id) for arm_id in self.arms.keys()]
		if any([c[0] < self.stop_after for c in counts]):
			choices = [c[1] for c in counts if c[0] < self.stop_after]
			return np.random.choice(choices)
		else:
			means = [(self.get_expected_value(arm_id), arm_id) for arm_id in self.arms.keys()]
			return sorted(means)[-1][1]

class Greedy(Bandit):    
	def __init__(self, n_arms):
		super().__init__(n_arms)

	def choose_arm(self):
		means = [(self.get_expected_value(arm_id), arm_id) for arm_id in self.arms.keys()]
		return sorted(means)[-1][1]

class EpsilonGreedy(Bandit):
	def __init__(self, n_arms, epsilon=0.1):
		super().__init__(n_arms)
		self.epsilon = epsilon

	def choose_arm(self):
		if np.random.random() > self.epsilon:
			means = [(self.get_expected_value(arm_id), arm_id) for arm_id in self.arms.keys()]
			return sorted(means)[-1][1]
		else:
			return np.random.choice(list(self.arms.keys()))

class EpsilonDecreasing(EpsilonGreedy):
	def __init__(self, n_arms, epsilon = 0.1, ep_factor = 0.99):
		super().__init__(n_arms, epsilon)
		assert (ep_factor < 1.0 and ep_factor > 0.0)
		self.ep_factor = ep_factor

	def choose_arm(self):
		self.epsilon *= self.ep_factor
		if np.random.random() > self.epsilon:
			means = [(self.get_expected_value(arm_id), arm_id) for arm_id in self.arms.keys()]
			return sorted(means)[-1][1]
		else:
			return np.random.choice(list(self.arms.keys()))
		
class UCB(Bandit):
	def __init__(self, n_arms = 2, delta = 0.015):
		super().__init__(n_arms)
		self.delta = delta

	def get_variance(self, arm_id):
		T = float(sum([len(self.arms[i]['observations']) for i in self.arms.keys()]))
		n = float(len(self.arms[arm_id]['observations']))
		return self.delta*1.96*sqrt(2*T/n)

	def choose_arm(self):
		ucbs = []
		for arm_id in self.arms.keys():
			mean = self.get_expected_value(arm_id)
			variance = self.get_variance(arm_id)
			ucbs.append((mean+variance, arm_id))
		return sorted(ucbs)[-1][1]

class Bayesian(Bandit):
	def __init__(self, n_arms = 2, a = 1, b = 1):
		self.n = n_arms
		self.arms = {}
		for i in range(self.n):
			self.arms[i] = {}
			self.arms[i]['a'] = a
			self.arms[i]['b'] = b
		self.default_alpha = a
		self.default_beta = b

	def choose_arm(self):
		#rvs = []
		#for arm_id in self.arms.keys():
		#    rvs.append((np.random.beta(self.arms[arm_id]['a'], self.arms[arm_id]['b']), arm_id))
		rvs = [(np.random.beta(self.arms[i]['a'], self.arms[i]['b']), i) for i in self.arms.keys()]
		return sorted(rvs)[-1][1]

	def update(self, arm_id, reward):
		if reward == 0:
			self.arms[arm_id]['b'] += 1.0
		elif reward == 1:
			self.arms[arm_id]['a'] += 1.0

	def add_arm(self, arm_id = None):
		if not arm_id:
			self.n += 1
			arm_id = self.n
		else:
			arm_id = arm_id
		self.arms[arm_id] = {}
		self.arms[arm_id]['a'] = self.default_alpha
		self.arms[arm_id]['b'] = self.default_beta
		return arm_id

	def reset_arm(self, arm_id):
		self.arms[arm_id]['a'] = self.default_alpha
		self.arms[arm_id]['b'] = self.default_beta

	def get_expected_value(self, arm_id):
		a = self.arms[arm_id]['a']
		b = self.arms[arm_id]['b']
		expected_value = a/(a+b)
		return expected_value

	def get_variance(self, arm_id):
		a = self.arms[arm_id]['a']
		b = self.arms[arm_id]['b']
		variance = (a*b)/(((a+b)**2)*(a+b+1))
		return variance

class Bayesian2(Bayesian):
	def __init__(self, n_arms = 2, a = 1, b = 1, C = 50):
		super().__init__(n_arms, a, b)
		self.C = C
		
	def choose_arm(self):
		rvs = [(np.random.beta(self.arms[i]['a'], self.arms[i]['b']), i) for i in self.arms.keys()]
		return sorted(rvs)[-1][1]

	def update(self, arm_id, reward):
		C = self.C
		if reward == 0:
			if self.arms[arm_id]['a'] + self.arms[arm_id]['b'] > C:
				self.arms[arm_id]['b'] += 1.0
			else:
				self.arms[arm_id]['a'] *= C/float(C+1)
				self.arms[arm_id]['b'] = (self.arms[arm_id]['b']+1)*C/float(C+1)

		elif reward == 1:
			if self.arms[arm_id]['a'] + self.arms[arm_id]['b'] <= C:
				self.arms[arm_id]['a'] += 1.0
			else:
				self.arms[arm_id]['a'] = (self.arms[arm_id]['a']+1)*C/float(C+1)
				self.arms[arm_id]['b'] *= C/float(C+1)

	def add_arm(self, arm_id = None):
		if not arm_id:
			self.n += 1
			arm_id = self.n
		else:
			arm_id = arm_id
		self.arms[arm_id] = {}
		self.arms[arm_id]['a'] = self.default_alpha
		self.arms[arm_id]['b'] = self.default_beta
		return arm_id

	def reset_arm(self, arm_id):
		self.arms[arm_id]['a'] = self.default_alpha
		self.arms[arm_id]['b'] = self.default_beta

	def get_expected_value(self, arm_id):
		a = self.arms[arm_id]['a']
		b = self.arms[arm_id]['b']
		expected_value = a/(a+b)
		return expected_value

	def get_variance(self, arm_id):
		a = self.arms[arm_id]['a']
		b = self.arms[arm_id]['b']
		variance = (a*b)/(((a+b)**2)*(a+b+1))
		return variance

if __name__ == '__main__':
	b = Bayesian2()
	print(b.get_expected_values_all())
	print(b.get_variances_all())
	b.update(0,1)
	print(b.get_expected_values_all())
	print(b.get_variances_all())
	b.update(0,1)
	print(b.get_expected_values_all())
	print(b.get_variances_all())
	b.update(1,1)
	print(b.get_expected_values_all())
	print(b.get_variances_all())
	b.update(0,1)
	print(b.get_expected_values_all())
	print(b.get_variances_all())
	print(b.arms[0])
	print(b.arms[1])
