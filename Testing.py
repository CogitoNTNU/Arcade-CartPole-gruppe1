import gym
import numpy as np
import random

# Set the percent you want to explore    
# Initialize q-table values to 0
discrete = 100
Q = np.zeros((100, 2))
epsilon = 0.2
gamma = 0.8
lr = 0.1
total = 0


angleTable = np.linspace(-0.20943951, 0.20943951, num = 100)

def getState(angle):
    for roundedAngle in angleTable:
        if roundedAngle > angle:
            index = np.where(angleTable == roundedAngle)[0][0]
            return index

def betterAction(angle):
    #Finn en bedre action enn noe tilfeldig.
    choices = Q[getState(angle)]
    indexOfChoice = np.argmax(choices)
    return indexOfChoice

env = gym.make('CartPole-v0')
for i_episode in range(1000):
    observation = env.reset()
    rew = 0
    for t in range(2000):
        #env.render()
        angle = observation[2]
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:  
            action = betterAction(angle)
        observation, reward, done, info = env.step(action)
        rew += reward

        # Update q values
        state = getState(angle)
        newAngle = observation[2]
        new_state = getState(newAngle)
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        
        if i_episode > 950:
            env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        total += rew
print(total)
print("Ti første \n", Q[0:10], "\n")
print("50-55 \n", Q[50:55], "\n")
print("Ti siste \n", Q[-10:])
env.close()
