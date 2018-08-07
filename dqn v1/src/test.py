from Map import Map
from DQN import DeepQNetwork
from compare import Compare
from findDestinations import findDestinations
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import random
# Global variables
# this script doesn't work on cluster because RL shouldn't be a global variable, but this should be transformed into a class
# using self.RL, as seen in the newer versions.

MIN_VALUE = -10000
IMG_SIZE = 300
action_space = ['u', 'd', 'l', 'r','ur','rd','ld','ul']
n_actions = len(action_space)
n_features = 2
RL = DeepQNetwork(n_actions, n_features,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.9,
                    replace_target_iter=200,
                    memory_size=2000,
                    output_graph=False,
                    testing=True
                    )

def init_map(filename):
    map_file = np.loadtxt('map.txt',dtype=int)
    # bounding negative values for keeping it in bounds
    map_file[0,:] = MIN_VALUE
    map_file[:,0] = MIN_VALUE
    map_file[:,len(map_file)-1]=MIN_VALUE
    map_file[len(map_file)-1,:]=MIN_VALUE
    return map_file

def run_testcase(test_samples):
    map_file = init_map('map.txt')
    case = 1
    for sample in test_samples:
        print("tested on: ",case)
        start = sample[0]
        end = sample[1]
        filename = str(case)
        case +=1
        # UAV map emulation
        env = Map(start,end,filename,map_file)
        total_epochs = 300
        run_map(filename,env,True, total_epochs)   
        RL.plot_cost()
        Compare(filename)

def run_map(i,env,testing,epochs):
    step = 0
    s = []
    for episode in range(epochs):
        print("starting epoch ", episode)
        # initial observation
        observation = env.reset(str(episode))
        count = 0
        while True:
            count += 1
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        s.append(count)
    
    plt.plot(np.arange(len(s)), s)
    plt.ylabel('points to goal')
    plt.xlabel('training steps')
    
    folder = "../DQN_path/graphs/"
    figname = folder + i + "_test_" + "_figPtsv1.png"                
    plt.savefig(figname)
    plt.clf()

if __name__ == "__main__":
    test_samples = []
    start_table = dict()
    end_table = dict()
    
    for case_number in range(1,15):
        filename = str(case_number)
        # find destinations in folder starting_... and ending_...
        finder = findDestinations(filename)
        end = finder.returnDestination()
        start = finder.returnStarting()
        test_samples.append((start,end))
        start_table[start[0]] = start[1]
        end_table[end[0]] = end[1]

    # Testing Time keeping
    time_file = "testTime.txt"
    start = time.time()

    # Test on 15 samples
    run_testcase(test_samples)

    # Time keeping
    f = open(time_file,"w+")
    total_time = (time.time() - start)/(60*15) # print avg minutes to test
    f.write(str(total_time))
    f.close()
