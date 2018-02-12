# implementation of a q -learnign network using nn with numpy
# goals - to understand q-learnign algo
# implementation for the algo here - http://mnemstudio.org/path-finding-q-learning-tutorial.htm

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random


#when bot reaches the expected state reqrd is 1 or else -1
# goal is to slect the object/action pair with the highest reward

# R matrix
R = np.matrix([[-1, -1, -1, -1, 1, -1],
               [-1, -1, -1, 1, -1, -1],
               [-1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, 1, -1],
               [-1, -1, -1, -1, -1, 1],
               [-1, -1, -1, -1, -1, 1]])




# Q matrix
Q = np.matrix(np.zeros([6, 6]))

# Gamma (learning parameter).
gamma = 0.8

# Initial state. (Usually to be chosen at random)
initial_state = 1
w = 2*np.random.random((6,1)) - 1  # random weights for each action

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]
    #av_act = np.where(current_state_row >= 0)[1]
    av_act = np.where(current_state_row)[1]
    #print("available action : " , av_act)
    return av_act



# This function chooses at random which action to be performed within the range
# of all the available actions.
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range, 1))     #  np.random.choice(available_act, 1 - returns an array of size 1
    #print("next_action: ", next_action)
    return next_action

def Qpredict(action,reward,w):
    Qpredict = reward*w[action,0]
    return Qpredict



# This function updates the Q matrix according to the path selected and the Q
# learning algorithm
#def update(current_state, action, gamma):
#    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
#    print("max_index : ",max_index)
#    print("max_index.shape : ", max_index.shape)
#
#    if max_index.shape[0] > 1:
#        max_index = int(np.random.choice(max_index, size=1))
#        print("max_index after if : ",max_index)
#    else:
#        max_index = int(max_index)
#        print("max_index after else : ",max_index)
#
#    max_value = Q[action, max_index]
#    print("max_value: ",max_value)
#
#    # Q learning formula
#    Q[current_state, action] = R[current_state, action] + gamma * max_value
#    print("Q[current_state, action] :" ,Q[current_state, action] )

def update(Qtarget,current_state, action):
    Q[current_state, action] =  Qtarget
    print("Q[current_state, action] :" ,Q)


## -------------------------------------------------------------------------------
## Training

# Train over 10 000 iterations. (Re-iterate the process above).

#reward = 0


#current_state = np.random.randint(0, int(Q.shape[0]))

# todo - to be extracted randomly
current_state = 1

print("initial state : ",current_state)
qpredict = 0
qtarget = 0

for i in range(6):
    print("inside for loop now :")

    print("current_state : ",current_state)
    available_act = available_actions(current_state)
    print("available_act : ", available_act)


    while True:
        action = sample_next_action(available_act)

        if R[current_state,action] > 0:
            print("action :", action)
            #break
            reward = R[current_state,action]


            print("reward : ",reward)
            qpredict = Qpredict(action,reward,w)
            qtarget =  reward + (gamma*qpredict)

            # updatingthe q table for representation purpose only when using nn u dont need this
            update(qtarget,current_state, action)

            current_state = action
            break
        else:
          loss = qtarget - qpredict
          adjustment = R[current_state,action] * (loss * qpredict)
          w += adjustment

    #current_state = action
    print("weights are : ",w)

    #updatingthe q table for representation purpose only when using nn u dont need this
    #update(qtarget,current_state, action)











    #update(current_state, action, gamma)
    #print('-----------------')
