import numpy as np
import os
UNIT = 1   # pixels
MAZE_H = 300  # grid height
MAZE_W = 300  # grid width

class Map(object):
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self._build_map()

    def _build_map(self):
        # create origin
        origin = np.array([118, 85]) # starting position
        
        self.maze_arr = np.loadtxt('map.txt',dtype=int)#.reshape((512,512))

        # create oval # ending position
        oval_center = np.array([111,124]) # afterwards replace these with reading images
        # Set reward
        self.oval = [
            oval_center[0] - 5, oval_center[1] - 5,
            oval_center[0] + 5, oval_center[1] + 5]

        self.maze_arr[oval_center[0]-5:oval_center[0]+5,oval_center[1]-5:oval_center[1]+5] = 1000
        
        # create black rect aka plane
        self.plane = [origin[0] - 5, origin[1] - 5,
             origin[0] + 5, origin[1] + 5]
       
    def reset(self,episode):
        origin = np.array([111,124]) # starting position
        self.plane = [origin[0] - 5, origin[1] - 5,
             origin[0] + 5, origin[1] + 5]
        self.episode = episode
        observation = (np.array(self.plane[:2]) - np.array(self.oval)[:2])/(MAZE_H*UNIT)
        self.positions_visited = []
        self.positions_visited.append(self.plane)
       
        if(int(episode)<10 or int(episode)>290):
            self.file_path = os.getcwd() + "/pts"
            if(not os.path.exists(self.file_path)):
                os.mkdir(self.file_path)
            f = open(self.file_path + "/" + self.episode + ".txt","w+")
            f.write(str(self.plane) + "\n")
            f.close()

        return observation

    def step(self, action):
        s = self.plane
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

        self.plane[0] += base_action[0]
        self.plane[2] += base_action[0]
        self.plane[1] += base_action[1]
        self.plane[3] += base_action[1]
        
        next_coords = self.plane
        self.positions_visited.append(self.plane)
       
        if(int(self.episode) < 10 or int(self.episode) > 290):
            f = open(self.file_path + "/"+self.episode +".txt","a+")
            f.write(str(self.plane)+"\n")
            f.close()
        # reward function
        goal_coords = self.oval
        if next_coords == goal_coords:
            reward = 10 + self.euclidean(next_coords,goal_coords)            
            # only stop when goal is reached
            done = True
        else:
            curr_x = int((next_coords[0] + next_coords[2]) // 2)
            curr_y = int((next_coords[1] + next_coords[3]) // 2)

            # Repeating values for padding
            if curr_x >= 300:
                curr_x = 299
            if curr_y >= 300:
                curr_y = 299

            table_lookup = self.maze_arr[curr_x,curr_y]
            
            if(table_lookup == 255):
                table_lookup = -5
            if(table_lookup == 1):
                table_lookup = -1
                
            reward = table_lookup + self.euclidean(next_coords,goal_coords)
            done = False
       
        s_ = (np.array(next_coords[:2]) - np.array(self.oval[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def euclidean(self,coord,goal_coord):
        curr_x = (coord[0] + coord[2]) // 2
        curr_y = (coord[1] + coord[3]) // 2
        goal_x = (goal_coord[0] + goal_coord[2]) // 2
        goal_y = (goal_coord[1] + goal_coord[3]) // 2
        dist = ( (goal_x - curr_x)**2 + (goal_y - curr_y)**2 )**.5
        return 1/(dist + 1.0)

