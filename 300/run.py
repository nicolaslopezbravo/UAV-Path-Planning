from Map import Map
from DQN import DeepQNetwork
import matplotlib.pyplot as plt
import time
import numpy as np

def run_map():
    step = 0
    total_time = 0
    start = time.time()
    s = []
    for episode in range(300):
        # initial observation
        observation = env.reset()
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
    plt.savefig("figPtsv1.png")

    total_time = start - time.time()
    f = open("trainTime.txt","w+")
    f.write(total_time)
    f.close()
    print('Finished')

if __name__ == "__main__":
    # maze game
    env = Map()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    run_map()
    
RL.plot_cost()