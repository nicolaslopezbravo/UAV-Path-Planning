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

def run_training(training_samples):
    # Train over multiple instances
    map_file = np.loadtxt('map.txt',dtype=int)
    # bounding negative values for keeping it in bounds
    map_file[0,:] = MIN_VALUE
    map_file[:,0] = MIN_VALUE
    map_file[:,len(map_file)-1]=MIN_VALUE
    map_file[len(map_file)-1,:]=MIN_VALUE

    for sample_x in range(training_samples):
        start = [random.randint(1,IMG_SIZE-1),random.randint(1,IMG_SIZE-1)]
        end = [random.randint(1,IMG_SIZE-1),random.randint(1,IMG_SIZE-1)]

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
                RL.learn(done)

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
    plt.savefig("../DQN_path/graphs/" + i + "_figPtsv1.png" )
    plt.clf()

if __name__ == "__main__":

    # Training Time keeping
    total_time = 0
    start = time.time()
    print("about to train")

    # train on 150 samples
    run_training(150)

    # Training Time keeping
    total_time = (time.time() - start)/60 # print minutes to train on 100 samples
    time_file = "trainTime.txt"
    f = open(time_file,"w+")
    f.write(str(total_time))
    f.close()