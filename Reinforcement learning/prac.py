import random
import torch
import torch.nn.functional as F
action= 4
#
# myaction=random.randint(1, action)
# print(myaction)
# #
# # expected=torch.tensor([1.1088, 1.1231, 1.3270, 1.5933, 1.0418, 1.2354, 0.9954, 1.0418, 1.4347,
# #         0.9416, 1.0450, 1.4721, 1.1185, 1.1303, 1.1096],)
# #        grad_fn=<ThAddBackward>)
# #
# # state_vale=torch.tensor([-0.1566, -0.0804,  0.1751,  0.1374, -0.0302, -0.0302, -0.0302, -0.0416,
# #          0.5946, -0.0939,  0.1188,  0.5774,  0.1284,  0.1384,  0.0552],
#
# expected=torch.tensor([1.1088, 1.1231, 1.3270, 1.5933, 1.0418, 1.2354, 0.9954, 1.0418, 1.4347,
#         0.9416, 1.0450, 1.4721, 1.1185, 1.1303, 1.1096], requires_grad=True)
#
# state_vale=torch.tensor([-0.1566, -0.0804,  0.1751,  0.1374, -0.0302, -0.0302, -0.0302, -0.0416,
#          0.5946, -0.0939,  0.1188,  0.5774,  0.1284,  0.1384,  0.0552])
#
# loss=F.smooth_l1_loss(expected,state_vale)
# print(loss)
#
# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
#
# x = Variable(torch.randn(2, 2), requires_grad=True)
# t = Variable(torch.randn(2, 2), requires_grad=False)
# print(F.smooth_l1_loss(x, t, reduce=True))
#

import numpy as np

def return_state_utility(v, T, u, reward, gamma):
    """Return the state utility.

    @param v the state vector
    @param T transition matrix
    @param u utility vector
    @param reward for that state
    @param gamma discount factor
    @return the utility of the state
    """
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return reward + gamma * np.max(action_array)

def main():
    #Starting state vector
    #The agent starts from (1, 1)
    v = np.array([[0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0]])

    #Transition matrix loaded from file
    #(It is too big to write here)
    T = np.load("T.npy")
    # print(T)

    #Utility vector
    u = np.array([[0.812, 0.868, 0.918,   1.0,
                   0.762,   0.0, 0.660,  -1.0,
                   0.705, 0.655, 0.611, 0.388]])

    #Defining the reward for state (1,1)
    reward = -0.04
    #Assuming that the discount factor is equal to 1.0
    gamma = 1

    #Use the Bellman equation to find the utility of state (1,1)
    utility_11 = return_state_utility(v, T, u, reward, gamma)
    print("Utility of state (1,1): " + str(utility_11))

if __name__ == "__main__":
    main()


import numpy as np

def return_state_utility(v, T, u, reward, gamma):
    """Return the state utility.

    @param v the value vector
    @param T transition matrix
    @param u utility vector
    @param reward for that state
    @param gamma discount factor
    @return the utility of the state
    """
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return reward + gamma * np.max(action_array)

def main():
    #Change as you want
    tot_states = 12
    gamma = 0.999 #Discount factor
    iteration = 0 #Iteration counter
    epsilon = 0.01 #Stopping criteria small value

    #List containing the data for each iteation
    graph_list = list()

    #Transition matrix loaded from file (It is too big to write here)
    T = np.load("T.npy")

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])
    u1 = np.array([0.0, 0.0, 0.0,  0.0,
                    0.0, 0.0, 0.0,  0.0,
                    0.0, 0.0, 0.0,  0.0])

    while True:
        delta = 0
        u = u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(tot_states):
            reward = r[s]
            v = np.zeros((1,tot_states))
            v[0,s] = 1.0
            u1[s] = return_state_utility(v, T, u, reward, gamma)
            delta = max(delta, np.abs(u1[s] - u[s])) #Stopping criteria
        if delta < epsilon * (1 - gamma) / gamma:
                print("=================== FINAL RESULT ==================")
                print("Iterations: " + str(iteration))
                print("Delta: " + str(delta))
                print("Gamma: " + str(gamma))
                print("Epsilon: " + str(epsilon))
                print("===================================================")
                print(u[0:4])
                print(u[4:8])
                print(u[8:12])
                print("===================================================")
                break

if __name__ == "__main__":
    main()


import numpy as np

def return_policy_evaluation(p, u, r, T, gamma):
  """Return the policy utility.

  @param p policy vector
  @param u utility vector
  @param r reward vector
  @param T transition matrix
  @param gamma discount factor
  @return the utility vector u
  """
  for s in range(12):
    if not np.isnan(p[s]):
      v = np.zeros((1,12))
      v[0,s] = 1.0
      action = int(p[s])
      u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
  return u

def return_expected_action(u, T, v):
    """Return the expected action.

    It returns an action based on the
    expected utility of doing a in state s,
    according to T and u. This action is
    the one that maximize the expected
    utility.
    @param u utility vector
    @param T transition matrix
    @param v starting vector
    @return expected action (int)
    """
    actions_array = np.zeros(4)
    for action in range(4):
       #Expected utility of doing a in state s, according to T and u.
       actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return np.argmax(actions_array)

def print_policy(p, shape):
    """Printing utility.

    Print the policy actions using symbols:
    ^, v, <, > up, down, left, right
    * terminal states
    # obstacles
    """
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " *  "
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

if __name__ == "__main__":
    main()


def main():
    gamma = 0.999
    epsilon = 0.0001
    iteration = 0
    T = np.load("T.npy")
    #Generate the first policy randomly
    # NaN=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1
    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                  0.0, 0.0, 0.0,  0.0,
                  0.0, 0.0, 0.0,  0.0])
    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        #1- Policy evaluation
        u_0 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        #Stopping criteria
        delta = np.absolute(u - u_0).max()
        if delta < epsilon * (1 - gamma) / gamma: break
        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1,12))
                v[0,s] = 1.0
                #2- Policy improvement
                a = return_expected_action(u, T, v)
                if a != p[s]: p[s] = a
        print_policy(p, shape=(3,4))

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")

if __name__ == "__main__":
    main()
