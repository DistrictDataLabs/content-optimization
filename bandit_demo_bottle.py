from bottle import route, run, template
import numpy as np
from bandit_classes import Bayesian2

@route('/')
def index():
	c = get_next_button_color()
	return template('call_to_action.tpl', button_color=c, 
		arm_index=current_arm_index)

@route('/cta', method='GET')
def thanks():
	bandit.update(current_arm_index, reward=1)
	arm_data = get_bandit_arm_data()
	return template('bandit_state.tpl', message='Thanks for clicking!',
		 arm_data=arm_data)

@route('/no', method='GET')
def no():
	bandit.update(current_arm_index, reward=0)
	arm_data = get_bandit_arm_data()
	return template('bandit_state.tpl', message='Thanks anyway!',
		 arm_data=arm_data)

def get_next_button_color():
	global current_arm_index
	current_arm_index = bandit.choose_arm()
	return colors[current_arm_index]
	#return colors[current_arm_index]

def get_bandit_arm_data():
	#arm_data = bandit.get_arm_data()
	means = bandit.get_expected_values_all()
	#variances = bandit.get_variances_all()
	#arm_data = zip(means, variances)
	arm_data = list(zip(range(len(colors)), means))
	return arm_data

current_arm_index = -1
colors = ['red', 'green', 'blue']
bandit = Bayesian2(n_arms=len(colors))
run(host='localhost', port=8089, reloader=True)
