import gym
import gym_maze
import time
import numpy as np
import random
env = gym.make("maze-sample-3x3-v0")
env.reset()
qtable = np.zeros((3,3,4))
arr_action = ['N','E','S','W']
epsilon = 0.9
epsilon_decay = 0.99
gammar = 0.9
alpha = 0.9
for i in range(1000):
    cum_reward = 0
    done = False
    state = [0, 0]
    cum_reward = 0
    step = 0
    env.reset()
    while not done:
        step = step + 1
        time.sleep(1)
        if np.random.rand() <= epsilon:
            a = random.randint(0,3)
        else:
            a = int(np.argmax(qtable[state[0]][state[1]]))
        newstate, reward, done, dc = env.step(a)  # take a random action
        env.render()
        cum_reward = cum_reward + reward
        print(state,type(state), arr_action[a], newstate, reward, done,cum_reward)
        if done:
            qtable[state[0]][state[1]][a] = reward
            epsilon = epsilon * epsilon_decay
        else:
            qsa = qtable[state[0]][state[1]][a]
            amax = np.amax(qtable[newstate[0]][newstate[1]])
            qtable[state[0]][state[1]][a] = qsa + alpha*(reward + gammar*amax - qsa)
        prev_state = state
        state = newstate
    print("Done",i,"Step",step,'CumReward',cum_reward,"epsilon",epsilon)
