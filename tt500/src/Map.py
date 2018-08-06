import numpy as np
import os
UNIT = 1   # pixels
MAZE_H = 300  # grid height
MAZE_W = 300  # grid width
MAX_VAL= 1000
MIN_VAL=-10000
ROAD_REWARD = -1
BUILDING_REWARD = 1

class Map(object):
    def __init__(self,origin,destination,iteration,map_file):
        self.action_space = ['u', 'd', 'l', 'r','ur','rd','ld','ul'] #accounting for diagonals
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.origin = np.array(origin) # starting position [118, 85]
        self.maze_arr = map_file
        self.destination = destination
        self.benchmark_distance = ((self.destination[0] - self.origin[0])**2 + (self.destination[1] - self.origin[1])**2 )**.5
        self.maze_arr[self.destination[0], self.destination[1]] = 1000
        self.plane = [self.origin[0], self.origin[1]]
        self.iteration = str(iteration)
       
    def reset(self,episode):
        self.plane = [self.origin[0], self.origin[1]]
        self.episode = episode # episode means epoch here
        observation = (np.array(self.plane) - np.array(self.destination))/(MAZE_H*UNIT) # observation is the distance from the target 
       
        if(int(episode)>298): # or int(episode)>290):
            f = open("../DQN_path/"+self.iteration + ".txt","w+")
            f.write(str(self.plane) + "\n")
            f.close()

        return observation

    def step(self, action):
        s = self.plane # current location -- self.origin initially 
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 4: #up right
            if s[1] > UNIT:
                base_action[1] -= UNIT
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 5: #right down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 6: #left down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 7: #up left
            if s[1] > UNIT:
                base_action[1] -= UNIT
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.plane[0] += base_action[0]
        self.plane[1] += base_action[1]
        
        next_coords = self.plane # updated state 
       
        if(int(self.episode) > 298): # only write the last episode
            f = open("../DQN_path/"+self.iteration +".txt","a+")
            f.write(str(self.plane)+"\n")
            f.close()
        # reward function
        goal_coords = self.destination
        if next_coords == goal_coords:
            reward = MAX_VAL            
            # only stop when goal is reached
            done = True
        else:
            curr_x = next_coords[0]
            curr_y = next_coords[1]

            table_lookup = self.maze_arr[curr_x,curr_y] # read the image rewards from the combined image
            
            if(table_lookup == 255):
                table_lookup = ROAD_REWARD  # for roads
            if(table_lookup == 1):
                table_lookup = BUILDING_REWARD  # for buildings

            dist = self.euclidean(next_coords,goal_coords)
            if(((-1)*dist) < self.benchmark_distance):
                dist = 2.77**(dist)
                
            reward = table_lookup + (dist/(MAZE_H*UNIT)) # do some cases with h*w
            done = False
            # if it diverges stop it
            
            if(table_lookup == MIN_VAL):
                done = True       
        s_ = (np.array(next_coords) - np.array(self.destination))/(MAZE_H*UNIT)
        return s_, reward, done

    def euclidean(self,coord,goal_coord):
        curr_x = coord[0]
        curr_y = coord[1]
        goal_x = goal_coord[0]
        goal_y = goal_coord[1]
        dist = ( (goal_x - curr_x)**2 + (goal_y - curr_y)**2 )**.5
        return -dist

