import numpy as np
import random
from collections import defaultdict
from environment import Env

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.8  # High learning rate to quickly adjust
        self.discount_factor = 0.99  # Prioritize future rewards
        self.epsilon = 0.1  # Start with low exploration
        self.epsilon_min = 0.01  # Very low minimum exploration after success
        self.epsilon_decay = 0.99  # Decay epsilon slowly
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.goal_reward_base = 100  # Base reward for reaching the goal
        self.time_penalty = -1  # Penalty per time step to encourage speed

    # Update Q-function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # Bellman equation update
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    def get_action(self, state):
        # Choose action with epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # Explore
        else:
            return self.arg_max(self.q_table[state])  # Exploit

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
        # Gradually reduce exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        steps_to_goal = 0

        # Limit steps to encourage faster success
        while True:
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)

            # Apply time penalty and compute dynamic goal reward
            reward += agent.time_penalty  # Penalize time steps
            steps_to_goal += 1

            if done:
                # Increase reward based on steps taken
                goal_reward = max(0, agent.goal_reward_base - steps_to_goal * 5)  # Decrease reward for longer routes
                reward += goal_reward
                total_reward += reward
                agent.learn(str(state), action, reward, str(next_state))
                break  # Exit loop on reaching the goal

            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

        # Decay exploration rate after each episode
        agent.decay_epsilon()

        # Print total reward per episode for monitoring
        print(f"Episode {episode+1}: Total Reward = {total_reward}")