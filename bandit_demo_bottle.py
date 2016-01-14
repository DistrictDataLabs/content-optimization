from bottle import route, run, template
import numpy as np
from BanditClasses import Bayesian2

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
	next_arm_index = bandit.choose_arms()[0]
	global current_arm_index
	current_arm_index = next_arm_index
	return colors[current_arm_index]

def get_bandit_arm_data():
	arm_data = bandit.get_arm_data()
	arm_data = list(zip(range(len(colors)), [e[0] for e in arm_data]))
	return arm_data

current_arm_index = -1
colors = ['red', 'green', 'blue']
bandit = Bayesian2(k=len(colors))
run(host='localhost', port=8089, reloader=True)
