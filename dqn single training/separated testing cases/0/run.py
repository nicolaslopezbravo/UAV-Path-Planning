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
TOTAL_EPOCH = 300
MIN_VALUE = -10000

def run_training(filename):
    # find destinations in folder starting_... and ending_...
    finder = findDestinations(filename)
    end = finder.returnDestination()
    start = finder.returnStarting()

    map_file = np.loadtxt('map.txt',dtype=int)

    # bounding negative values for keeping it in bounds
    map_file[0,:] = MIN_VALUE
    map_file[:,0] = MIN_VALUE
    map_file[:,len(map_file)-1]=MIN_VALUE
    map_file[len(map_file)-1,:]=MIN_VALUE
    
    # UAV map emulation
    env = Map(start,end,filename,map_file)
    self.RL = DeepQNetwork(env.n_actions, env.n_features,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.9,
                    replace_target_iter=200,
                    memory_size=2000,
                    output_graph=True,
                    iteration=filename
                    )
    run_map(filename,env,False)  
    self.RL.plot_cost()
    #compare to given results
    print("Finished training", filename)

def run_testcase(filename):
    # find destinations in folder starting_... and ending_...
    finder = findDestinations(filename)
    end = finder.returnDestination()
    start = finder.returnStarting()

    map_file = np.loadtxt('map.txt',dtype=int)

    # bounding negative values for keeping it in bounds
    map_file[0,:] = MIN_VALUE
    map_file[:,0] = MIN_VALUE
    map_file[:,len(map_file)-1]=MIN_VALUE
    map_file[len(map_file)-1,:]=MIN_VALUE
    
    # UAV map emulation
    env = Map(start,end,filename,map_file)
    # self.RL = DeepQNetwork(env.n_actions, env.n_features,
    #                learning_rate=0.01,
    #                reward_decay=0.9,
    #                e_greedy=0.9,
    #                replace_target_iter=200,
    #                memory_size=2000,
    #                output_graph=True,
    #                iteration=filename
    #                )
    run_map(filename,self.RL,env,True)  
    self.RL.plot_cost()
    compare = Compare(filename)
    #compare to given results
    print("Finished iteration", filename)

def run_map(i,env,testing):
    step = 0
    total_time = 0
    start = time.time()
    s = []
    for episode in range(TOTAL_EPOCH):
        print("starting epoch ", episode)
        # initial observation
        observation = env.reset(str(episode))
        count = 0
        while True:
            count += 1
            # RL choose action based on observation
            action = self.RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            self.RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0) and not testing:
                self.RL.learn()

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
    figname = i + "_figPtsv1.png"
    total_time = (time.time() - start)/60 # print minutes to train
    time_file = "trainTime.txt"
    if testing:
        figname = i + "_test_" + "_figPtsv1.png"
        time_file = "testTime.txt"
    plt.savefig(figname)

    
    if testing
    f = open(time_file,"w+")
    f.write(str(total_time))
    f.close()
    print('Finished')

if __name__ == "__main__":
    case_number = 1
    filename = str(case_number)
    run_training("2")
    run_testcase(filename)
    
