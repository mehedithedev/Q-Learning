# Q-Legends

import numpy as np
import random
from collections import defaultdict
from environment import Env

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.8
        self.discount_factor = 0.99
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.goal_reward_base = 100
        self.time_penalty = -1

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.arg_max(self.q_table[state])

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        steps_to_goal = 0

        while True:
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)

            reward += agent.time_penalty
            steps_to_goal += 1

            if done:
                goal_reward = max(0, agent.goal_reward_base - steps_to_goal * 5)
                reward += goal_reward
                total_reward += reward
                agent.learn(str(state), action, reward, str(next_state))
                break

            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

        agent.decay_epsilon()

        print(f"Episode {episode+1}: Total Reward = {total_reward}")