import numpy as np

class ReplayBuffer:
	def __init__(self):
		self.clear()

	def clear(self):
		self.states = []
		self.turns = []
		self.final_reward = None

	def store_transition(self, state, reward, turn):
		self.states.append(state)
		self.turns.append(turn)
		if reward != None:
			self.final_reward = reward