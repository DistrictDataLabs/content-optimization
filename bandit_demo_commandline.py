# bandit demo for the command line
# player vs bandit over T time steps for k arms

from bandit_classes import Bayesian
from bandit_data import *
import numpy as np

class BanditDemo(object):
	def __init__(self, bandit, data, T):
		self.bandit = bandit
		self.data = data
		self.T = T
		choices = set(['red', 'green', 'blue', 'yellow'])
		if data.shape[1] > len(choices):
			raise ValueError('in BanditDemo init():' + 
				'too many choices for this game!')
		else:
			self.choices = []
			for i in range(data.shape[1]):
				ch = np.random.choice(list(choices))
				self.choices.append(ch)
				choices.remove(ch)

	def handle_bandit_choice(self, t):
		arm = self.bandit.choose_arm()
		actual_prob = self.data[t][arm]
		user_action = np.random.random()
		if user_action <= actual_prob:
			self.bandit.update(arm, 1)
			return 1
		else:
			self.bandit.update(arm, 0)
			return 0

	def handle_player_choice(self, t):
		print('choices at time step', str(t)+':')
		for i in range(self.data.shape[1]):
			print(str(i) + ' (' + self.choices[i] + ')')
		choices = set(range(self.data.shape[1]))
		player_choice = -1
		while player_choice not in choices:
			player_choice = input('select from the choices above: ')
			try:
				player_choice = int(player_choice)
			except Exception as e:
				player_choice = -1
				print('error: please select a number from' + \
				'the choices given: ', ', '.join(str(c) for c in choices))
		print('your choice:', player_choice)
		actual_prob = self.data[t][player_choice]
		user_action = np.random.random()
		if user_action <= actual_prob:
			print('result: user clicked!\n')
			return 1
		else:
			print('result: user did not click.\n')
			return 0
		
	def wrapup(self, player_rewards, bandit_rewards):
		print('Game over!')
		print('Actual click-through rates for the colors:')
		for i in range(self.data.shape[1]):
			print (i, '(' +self.choices[i]+ '):', str(100*self.data[0][i])[:4], '%')
		print('\nAccuracy scores for you and the bandit:')
		print('You got', str(100*player_rewards.mean())[:4]+'% correct')
		print('The bandit got', str(100*bandit_rewards.mean())[:4]+'% correct')

	def run(self):
		res = []
		for i in range(self.T):
			pr = self.handle_player_choice(i)
			br = self.handle_bandit_choice(i)
			res.append((pr, br))
		res = np.array(res)
		player_rewards = res[:,0]
		bandit_rewards = res[:,1]
		self.wrapup(player_rewards, bandit_rewards)
		if player_rewards.mean() > bandit_rewards.mean():
			print('Congratulations! You beat the bandit!')
			return 1
		elif player_rewards.mean() < bandit_rewards.mean():
			print('Condolances. You were bested by the bandit.')
			return -1
		else:
			print('It\'s a tie!')
			return 0
		
if __name__ == '__main__':
	k, T = 2, 10
	while True:
		intro = '\nWelcome. You will be given a sequence of choices. ' + \
			'Your task is to pick the button color' + \
			' that results in the highest click-through rate. '+ \
			'Choose well and maximize your reward.\nThis game ' + \
			'has '+str(k)+' choices and '+str(T)+' time steps. ' + \
			'Good luck!\n'
		print(intro)
		eg = Bayesian(k)
		#bd = BanditData()
		data = gen_static_uniform(k, T)
		#print data
		demo = BanditDemo(eg, data, T)
		res = demo.run()
		choices = set(['y', 'n'])
		ch = '?'
		if res == 1: # player beat bandit
			while ch not in choices:
				ch = str(input('Would you like to play a harder version? (y/n): '))
			if ch == 'y':
				T += 10
				k = k+1
				if k > 4:
					k = 4
			else:
				break
		elif res == -1: # bandit beat player
			while ch not in choices:
				ch = str(input('Would you like to play again? (y/n): '))
			if ch == 'y':
				T -= 10
				if T < 10:
					T = 10
				k = k-1
				if k < 2:
					k = 2
			else:
				break
		else: # player tied bandit
			while ch not in choices:
				ch = str(input('Would you like to play again? (y/n): '))
			if ch == 'y':
				continue
			else:
				break
