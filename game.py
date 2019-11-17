import gym
import numpy as np
import json
import pickle
env = gym.make ('MountainCar-v0')
env.reset ()
DISCREET_OS_SIZE = [20]*len (env.observation_space.high)
LEARNING_RATE = .1
EPISODES = 2500
SHOW_EPISODE = 500
DISCOUNT = .95
descreet_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCREET_OS_SIZE
q_table = np.random.uniform (low = 2,high = 0, size = (DISCREET_OS_SIZE+[env.action_space.n]))
def get_descrete_state (state):
    discrete_state = (state-env.observation_space.low)/descreet_os_win_size
    return tuple (discrete_state.astype(np.int))
for episode in range (EPISODES):
    discrete_state = get_descrete_state(env.reset())
    done = False
    if episode % SHOW_EPISODE ==0:
        render = True
    else :
        render = False
    print (episode)
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward,done,_ = env.step (action)
        new_descrete_state = get_descrete_state(new_state)
        if render:
            env.render ()
        if not done:
            max_future_q = np.max(q_table[new_descrete_state])
            current_q = q_table[discrete_state+(action,)]
            new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)
            q_table[discrete_state + (action,)] =new_q
        elif new_state[0] >= env.goal_position:
            q_table [discrete_state+(action,)]=0
        discrete_state = new_descrete_state
env.close ()
with open("save.pickle", "wb") as f:
        pickle.dump(q_table, f)