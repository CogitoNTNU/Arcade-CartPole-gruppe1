# Position AND tip velocity
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

# Initialize q-table values to 0
discretization = 512
Q = np.zeros((20,discretization, env.action_space.n))    # (states_space, action_space)

# Initialize hyperparameters
gamma = 0.9         # Discount factor used to balance immediate and future reward.
lr = 0.1            # learning rate
epsilon = 1         # rate of exploration where 1 is completely random and 0 is deterministic

# Initialize list for plotting running mean
rewards = []
N = 100     # Number of episodes for the running mean

def running_mean(x, N):
    """function for creating list with running mean"""
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def get_state(cart_position, tip_speed):
    """Returns index of the given tip speed state in the Q-table"""
    positions = np.linspace(-5,5,num=5)
    states = np.linspace(-3,3,num=discretization)   # Discretized table of tip speeds. Range of 3 indicates possible speeds
    for rounded_cart_position in positions:
        if rounded_cart_position >= cart_position:
            cart_index = np.where(positions == rounded_cart_position)
            for rounded_tip_speed in states:
                if rounded_tip_speed >= tip_speed:
                    index = np.where(states == rounded_tip_speed)
                    index = index[0][0]
                    return cart_index, index

def better_action(cart_position, tip_speed):
    """Returns index of best action according to Q-table for a given tip speed"""
    possible_actions_in_state = Q[get_state(cart_position, tip_speed)]
    action_of_choice = np.argmax(possible_actions_in_state)
    return action_of_choice

def update_q(state, action, new_state, cart_pos, new_cart_pos, reward=1):
    Q[cart_pos, state, action] = Q[cart_pos, state, action] + lr * (reward + gamma * np.max(Q[new_cart_pos, new_state, :]) - Q[cart_pos, state, action])




for i_episode in range(6000):
    observation = env.reset()
    episode_reward = 0

    if 4000 < i_episode :
        # Drop exploration
        epsilon = 0

    
        
    for timestep in range(200):
        if i_episode > 4980:
            #env.render()
            pass
    

        tip_speed = observation[3]
        cart_position = observation[0]

        if random.uniform(0, 1) < epsilon:      # Chooses random action often if epsilon is high
            action = env.action_space.sample()

        else:
            action = better_action(cart_position, tip_speed)

        # Gather info about changes in environment 
        state = get_state(cart_position,tip_speed)
        observation, reward, done, info = env.step(action)
        new_tip_speed = observation[3]
        new_cart_pos = observation[0]
        new_state = get_state(new_cart_pos, new_tip_speed)

        # Update Q-table
        update_q(state, action, new_state, cart_position, new_cart_pos, reward)

        episode_reward += reward

        if done:
            #print("Episode finished after {} timesteps".format(timestep+1))
            break
    rewards.append(episode_reward)

env.close()

plt.plot(running_mean(rewards,N))
plt.show()