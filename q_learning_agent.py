# Q-Legends

import numpy as np
import random
from collections import defaultdict
from environment import Env

class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.8 # Increased learning rate from 0.01 to 0.8 to learn about the environment faster
        self.discount_factor = 0.99 # Increased discount factor from 0.9 to 0.99 to prioritize future rewards
        self.epsilon = 0.1 # the agent is allowed to explore 10% in any state and 90% to exploit the learned values
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        # Added new variables

        self.epsilon_decay = 0.99 # reduce the exploration rate over time
        self.epsilon_min = 0.01 # Ensures the agent never stops exploring
        self.goal_reward_base = 100 # gives the agent a base reward for reaching the goal
        self.time_penalty = -1 # Added a penalty for each step the agent takes to increase speed

        
    # update q function with sample <s, a, r, s'>
    # kept the same original code and logic
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
            return np.random.choice(self.actions) # returned action instantly instead of storing it in a variable to save memory
        else:
            # take action according to the q function table
            return self.arg_max(self.q_table[state]) # added state_action from original code as an argument to the arg_max function

    # did no changes here
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

    
    def decay_epsilon(self): # reduce the exploration rate over time to encourage more exploitation 
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # Ensures the agent never stops exploring with the epsilon_min value

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        total_reward = 0 # Initialized total reward for the episode to 0 to keep track of total rewards
        steps_to_goal = 0 # Initialized step counter to 0 to track steps taken to reach to the goal

        while True:
            #  removed env.render() to increase the speed of learning process instead of visual feedback

            # take action and proceed one step in the environment
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)

            reward += agent.time_penalty # Added a penalty for each step the agent takes to reach to the goal to increase speed
            steps_to_goal += 1 # Increment step counter by 1 to keep track of each step taken to reach to the goal

            if done:
                goal_reward = max(0, agent.goal_reward_base - steps_to_goal * 5) # implemented a dynamic goal reward based on the number of steps taken to reach to the goal
                reward += goal_reward # increment reward by goal reward
                total_reward += reward # update total reward with the reward
                agent.learn(str(state), action, reward, str(next_state))
                break

            # with sample <s,a,r,s'>, agent learns new q function
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

        agent.decay_epsilon() # calling decay_epsilon function to reduce the exploration rate over time

        print(f"Episode {episode+1}: Total Reward = {total_reward}") # print total reward for each episode for better understanding of the agent's performance