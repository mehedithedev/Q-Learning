# Q-Legends

import numpy as np
import random
from environment import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        # self.learning_rate = 0.1
        self.learning_rate = 0.5  # Higher learning rate for faster updates

        
        self.discount_factor = 0.9
        # self.epsilon = 0.05
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_decay = 0.99  # Aggressively decay exploration rate

        self.q_table = defaultdict(lambda: [5.0, 5.0, 5.0, 5.0])

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

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

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()

        while True:
            env.render()

            # take action and proceed one step in the environment
            compressed_state = hash(str(state))  # Hash state for compact lookup
            action = agent.get_action(compressed_state)

            next_state, reward, done = env.step(action)

            # with sample <s,a,r,s'>, agent learns new q function
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state
            env.print_value_all(agent.q_table)

            # if episode ends, then break
            if reward == 1:  # If goal is reached
                reward += 20  # Strong reward for faster learning
                break  # End episode early if goal is reached
            else:
                reward -= 0.1  # Penalize unnecessary actions

            agent.epsilon = max(0.01, agent.epsilon * 0.99)  # Faster decay for exploitation
            agent.learning_rate = max(0.001, agent.learning_rate * 0.999)  # Slower decay for continuous learning

