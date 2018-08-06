from DynamicMap import DynamicMap
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

class Agent(object):
    def __init__(self):
        print("starting job")
        action_space    = ['u', 'd', 'l', 'r','ur','rd','ld','ul']
        n_actions = len(action_space)
        n_features = 2
        self.RL = DeepQNetwork(n_actions, n_features,
                            learning_rate=0.01,
                            reward_decay=0.9,
                            e_greedy=0.9,
                            replace_target_iter=200,
                            memory_size=2000,
                            output_graph=False
                            )

        test_samples = []
        start_table = dict()
        end_table = dict()
        filename = "../test_destinations.txt"
        f = open(filename,"r")

        for line in f:
            nums = line.split(';')
            start = nums[0].split(',')
            end_ = nums[1].split(',')

            start = [0,0]
            end = [0,0]
            start[0] = int(start[0])
            start[1] = int(start[1])
            end[0] = int(end_[0])
            end[1] = int(end_[1])

            test_samples.append((start,end))
            start_table[start[0]] = start[1]
            end_table[end[0]] = end[1]

        # Training Time keeping
        total_time = 0
        start = time.time()
        
        # train on 100 samples
        self.run_training(1,start_table,end_table)

        # Training Time keeping
        total_time = (time.time() - start)/60 # print minutes to train on 100 samples
        time_file = "trainTime.txt"
        f = open(time_file,"w+")
        f.write(str(total_time))
        f.close()

        # Testing Time keeping
        time_file = "testTime.txt"
        start = time.time()

        # Test on 15 samples
        self.run_testcase(test_samples)

        # Time keeping
        f = open(time_file,"w+")
        total_time = (time.time() - start)/(60*15) # print avg minutes to test
        f.write(str(total_time))
        f.close()


    def run_training(self,training_samples,start_table,end_table):
        # Train over multiple instances
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
                #print("stuck here")

            total_epochs = 5

            # UAV map emulation
            env = DynamicMap(start,end,sample_x)
            
            self.run_map(str(sample_x),env,False, total_epochs) 

            print("Finished training", sample_x)
        print("done training")

        # Save model here

    def run_testcase(self,test_samples):
        case = 1
        for sample in test_samples:
            start = sample[0]
            end = sample[1]
            filename = str(case)
            # UAV map emulation
            env = DynamicMap(start,end,case)
            total_epochs = 300

            self.run_map(filename,env,True, total_epochs)   
            self.RL.plot_cost()
            Compare(filename)
            case +=1

    def run_map(self,i,env,testing,epochs):
        step = 0
        s = []
        print("about to run")
        for episode in range(epochs):
            print("starting epoch ", episode)
            # initial observation
            observation = env.reset(str(episode))
            count = 0
            while True:
                count += 1
                # self.RL choose action based on observation
                action = self.RL.choose_action(observation)
                print(action)
                # self.RL take action and get next observation and reward
                observation_, reward, done = env.step(action)

                self.RL.store_transition(observation, action, reward, observation_)

                if ((step > 200) and (step % 5 == 0)) and not testing:
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
        
        folder = "../DQN_path/graphs/"
        figname = folder + i + "_figPtsv1.png"

        if testing:
            figname = i + "_test_" + "_figPtsv1.png"
            
        plt.savefig(figname)
        plt.clf()

run = Agent()