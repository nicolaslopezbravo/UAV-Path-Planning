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
                    testing=False
                    )

def init_map(filename):
    map_file = np.loadtxt('map.txt',dtype=int)
    # bounding negative values for keeping it in bounds
    map_file[0,:] = MIN_VALUE
    map_file[:,0] = MIN_VALUE
    map_file[:,len(map_file)-1]=MIN_VALUE
    map_file[len(map_file)-1,:]=MIN_VALUE
    return map_file

def run_training(training_samples,start_table,end_table):
    # Train over multiple instances
    map_file = init_map('map.txt')

    for sample_x in range(training_samples):
        start = [random.randint(1,IMG_SIZE-1),random.randint(1,IMG_SIZE-1)]
        end = [random.randint(1,IMG_SIZE-1),random.randint(1,IMG_SIZE-1)]

        # query dictionary
        start_ = start_table.get(start[0],-1)
        end_ = end_table.get(end[0],-1)

        # ensure different than test cases
        while (start_ == start[1] and end_ == end[1]):
            start = [random.randint(1,IMG_SIZE-1),random.randint(1,IMG_SIZE-1)]
            end = [random.randint(1,IMG_SIZE-1),random.randint(1,IMG_SIZE-1)]
            start_ = start_table.get(start[0],-1)
            end_ = end_table.get(end[0],-1)

        total_epochs = 300

        # UAV map emulation
        env = Map(start,end,sample_x,map_file)
        run_map(str(sample_x),env, total_epochs) 

        print("Finished training", sample_x)
    print("done training")

    # Save model here

def run_map(i,env,epochs):
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

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

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
    figname = folder + i + "_figPtsv1.png"                
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

    # Training Time keeping
    total_time = 0
    start = time.time()
    
    # train on 100 samples
    run_training(100,start_table,end_table)

    # Training Time keeping
    total_time = (time.time() - start)/60 # print minutes to train on 100 samples
    time_file = "trainTime.txt"
    f = open(time_file,"w+")
    f.write(str(total_time))
    f.close()