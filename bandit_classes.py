from math import sqrt
import numpy as np
from numpy.random import choice

class AB(object):
    def __init__(self, n_arms, stop_after=30):
        self.n = n_arms
        self.stop_after = stop_after
        self.arms = {}
        for i in range(self.n):
            self.arms[i] = {}
            self.arms[i]['observations'] = []

    def choose_arm(self):
        counts = [(len(self.arms[arm_id]['observations']), arm_id) for arm_id in self.arms.keys()]
        if any([c[0] < self.stop_after for c in counts]):
            choices = [c[1] for c in counts if c[0] < self.stop_after]
            return choice(choices)
        else:
            means = [self.get_expected_value(arm_id), arm_id) for arm_id in self.arms.keys()]
            return np.max(means)[1]

    def reset_arm(self, arm_id):
        self.arms[i]['observations'] = []

    def update(self, arm_id, reward):
        self.arms[arm_id]['observations'].append(reward)

    def get_expected_value(self, arm_id):
        return np.mean(self.arms[arm_id]['observations'])

    def get_variance(self, arm_id):
        return np.var(self.arms[arm_id]['observations'])

    def get_expected_values_all(self):
        return list(map(lambda x: self.get_arm_data_expected_value(x), self.arms.keys()))

    def get_variances_all(self):
        return list(map(lambda x: self.get_variance(x), self.arms.keys()))

class Greedy(object):
    def __init__(self, n_arms,
                 default_count=1.0,
                 default_value=1.0):
        self.counts = np.zeros(n_arms)+default_count
        self.values = np.zeros(n_arms)+default_value
        self.n = n_arms
        self.default_count = default_count
        self.default_value = default_value

    def choose_arm(self):
        return np.argmax(self.values)

    def reset_arm(self, arm):
        self.counts[arm] = self.default_count
        self.values[arm] = self.default_value

    def update(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value
        new_value += (1 / float(n)) * reward
        self.values[arm] = new_value

    def get_arm_data(self):
        return zip(self.values, np.zeros(self.n))

class EpsilonGreedy(object):
    def __init__(self, n_arms, epsilon=0.1,
                 default_count=1.0,
                 default_value=1.0):
        self.counts = np.zeros(n_arms)+default_count
        self.values = np.zeros(n_arms)+default_value
        self.epsilon = epsilon
        self.n = n_arms
        self.default_count = default_count
        self.default_value = default_value

    def choose_arm(self):
        if np.random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return np.random.randint(self.n)

    def reset_arm(self, arm):
        self.counts[arm] = self.default_count
        self.values[arm] = self.default_value

    def update(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value
        new_value += (1 / float(n)) * reward
        self.values[arm] = new_value

    def get_arm_data(self):
        return zip(self.values, np.zeros(self.n))

class EpsilonDecreasing(object):
    def __init__(self, n_arms,
                 default_count=1.0,
                 default_value=1.0):
        self.counts = np.zeros(n_arms)+default_count
        self.values = np.zeros(n_arms)+default_value
        self.n = n_arms
        self.epsilon = (2*self.n)
        self.epsilon /= (self.counts.sum() + 2*self.n)
        self.default_count = default_count
        self.default_value = default_value

    def choose_arm(self):
        self.epsilon = (2*self.n)
        self.epsilon /= (self.counts.sum() + 2*self.n)
        if np.random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return np.random.randint(self.n)

    def reset_arm(self, arm):
        self.counts[arm] = self.default_count
        self.values[arm] = self.default_value

    def update(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value
        new_value += (1 / float(n)) * reward
        self.values[arm] = new_value

    def get_arm_data(self):
        return zip(self.values, np.zeros(self.n))

class UCB(object):
    def __init__(self, n_arms=2,
                 default_count=1.0,
                 default_value=1.0):
        self.counts = np.zeros(n_arms)+default_count
        self.values = np.zeros(n_arms)+default_value
        self.n = n_arms
        self.default_count = default_count
        self.default_value = default_value
        self.delta = 0.015

    def choose_arm(self):
        ucbs = [0.]*self.n
        T = self.counts.sum()
        for i in range(self.n):
            p = self.values[i]
            n = self.counts[i]
            ucb = p + self.delta*sqrt(2*T/n)
            ucbs[i] = ucb
        return np.argmax(ucbs)

    def reset_arm(self, arm):
        self.counts[arm] = self.default_count
        self.values[arm] = self.default_value

    def update(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value
        new_value += (1 / float(n)) * reward
        self.values[arm] = new_value

    def get_arm_data(self):
        ci = np.zeros(self.n)
        T = self.counts.sum()
        for i in range(self.n):
            p = self.values[i]
            n = self.counts[i]
            ci[i] = self.delta*sqrt(2*T/float(n))
        return zip(self.values, ci)

class Bayesian(object):
    def __init__(self, n_arms=2,
                 default_count=1.0,
                 default_value=1.0):
        self.counts = np.zeros(n_arms)+default_count
        self.values = np.zeros(n_arms)+default_value
        self.n = n_arms
        self.default_count = default_count
        self.default_value = default_value
        self.delta = 0.015

    def choose_arm(self):
        rvs = np.zeros(self.n)
        for i in range(self.n):
            rvs[i] = np.random.beta(self.default_value+self.values[i],
                               self.default_count+self.counts[i]-self.values[i])
        return np.argmax(rvs)

    def reset_arm(self, arm):
        self.counts[arm] = self.default_count
        self.values[arm] = self.default_value

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += reward

    def get_arm_data(self):
        expected_values = np.zeros(self.n)
        for i in range(self.n):
            a = self.default_value + self.values[i]
            b = self.default_count + self.counts[i] - self.values[i]
            expected_values[i] = a/(a+b)
        return zip(expected_values, np.zeros(self.n))

class Bayesian2(object):
    def __init__(self, k=2,
                 default_alpha=1.0,
                 default_beta=1.0,
                 C=20):
        self.k = k
        self.C = C
        self.default_alpha = default_alpha
        self.default_beta = default_beta
        self.arms = {}
        for i in range(self.k):
            self.arms[i] = {}
            self.arms[i]['a'] = self.default_alpha
            self.arms[i]['b'] = self.default_beta

    def choose_arms(self, k_arms=1):
        res = []
        for i in self.arms.keys():
            res.append((np.random.beta(self.arms[i]['a'], self.arms[i]['b']), i))
        return [e[1] for e in sorted(res, reverse=True)[:k_arms]]

    def choose_arm(self):
        return self.choose_arms(k_arms=1)[0]

    def add_arm(self):
        self.n += 1
        arm_id = self.k
        self.arms[arm_id] = {}
        self.arms[arm_id]['a'] = self.default_alpha
        self.arms[arm_id]['b'] = self.default_beta
        return arm_id

    def reset_arm(self, arm_id):
        self.arms[arm_id]['a'] = self.default_alpha
        self.arms[arm_id]['b'] = self.default_beta

    def remove_arm(self, arm_id):
        del self.arms[arm_id]

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

    def get_arm_data(self):
        expected_values = np.zeros(self.k)
        for i in range(self.k):
            a = self.arms[i]['a']
            b = self.arms[i]['b']
            expected_values[i] = a/(a+b)
        return list(zip(expected_values, np.zeros(self.k)))

class Bayesian3(object):
    def __init__(self, n_arms=2,
                 default_alpha=1.0,
                 default_beta=1.0,
                 C=20):
        self.n = n_arms
        self.C = C
        self.default_alpha = default_alpha
        self.default_beta = default_beta
        self.arms = {}
        for i in range(self.n):
            self.arms[i] = {}
            self.arms[i]['a'] = self.default_alpha
            self.arms[i]['b'] = self.default_beta

    def choose_arms(self, k_arms=1):
        res = []
        for i in self.arms.keys():
            res.append((np.random.beta(self.arms[i]['a'], self.arms[i]['b']), i))
        return [e[1] for e in sorted(res, reverse=True)[:k_arms]]

    def choose_arm(self):
        return self.choose_arms(k_arms=1)[0]

    def add_arm(self):
        self.n += 1
        arm_id = self.n
        self.arms[arm_id] = {}
        self.arms[arm_id]['a'] = self.default_alpha
        self.arms[arm_id]['b'] = self.default_beta
        return arm_id

    def reset_arm(self, arm_id):
        self.arms[arm_id]['a'] = self.default_alpha
        self.arms[arm_id]['b'] = self.default_beta

    def remove_arm(self, arm_id):
        del self.arms[arm_id]

    def update(self, arm_id, reward):
        if arm_id not in self.arms.keys():
            raise Exception('no such arm currently exists.')
        else:
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

    def get_expected_value(self, arm_id):
        a = self.arms[arm_id]['a']
        b = self.arms[arm_id]['b']
        expected_value = a/(a+b)
        return expected_value

    def get_variance(self, arm_id):
        a = self.arms[arm_id]['a']
        b = self.arms[arm_id]['b']
        variance = a*b/(((a+b)**2)*(a+b+1))
        return varience

    def get_expected_values_all(self):
        return list(map(lambda x: self.get_arm_data_expected_value(x), self.arms.keys()))

    def get_variances_all(self):
        return list(map(lambda x: self.get_variance(x), self.arms.keys()))

if __name__ == '__main__':
    b = Bayesian3()
    print(b.get_arm_data_all())
    b.update(0,1)
    print(b.get_arm_data_all())
    b.update(0,1)
    print(b.get_arm_data_all())
    b.update(1,1)
    print(b.get_arm_data_all())
    b.update(0,1)
    print(b.get_arm_data_all())
    print(b.arms[0])
    print(b.arms[1])
