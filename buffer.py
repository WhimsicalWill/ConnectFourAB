import numpy as np

class ReplayBuffer:
	def __init__(self):
		self.clear()

	def clear(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.is_terminals = []

	def store_transition(self, state, action, reward, is_terminal):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.is_terminals.append(is_terminal)