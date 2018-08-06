import numpy as np

UNIT = 5   # pixels
MAZE_H = 511  # grid height
MAZE_W = 511  # grid width

#class Maze(tk.Tk, object):
class Map(object):
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self._build_map()

    def _build_map(self):
        # create origin
        origin = np.array([326, 327]) # starting position
        self.maze_arr = np.loadtxt('map.txt',dtype=int)#.reshape((512,512))

        # create oval # ending position
        oval_center = np.array([398,399]) # afterwards replace these with reading images
        # Set reward
        self.oval = [
            oval_center[0] - 10, oval_center[1] - 10,
            oval_center[0] + 10, oval_center[1] + 10]

        self.maze_arr[oval_center[0]-10:oval_center[0]+10,oval_center[1]-10:oval_center[1]+10] = 1000
        
        # create black rect aka plane
        self.plane = [origin[0] - 10, origin[1] - 10,
             origin[0] + 10, origin[1] + 10]

    
    def reset(self):
        origin = np.array([326, 327]) # starting position
        self.plane = [origin[0] - 10, origin[1] - 10,
             origin[0] + 10, origin[1] + 10]
        observation = (np.array(self.plane[:2]) - np.array(self.oval)[:2])/(MAZE_H*UNIT)
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

        
        #self.canvas.move(self.plane, base_action[0], base_action[1])  # move agent
        self.plane[0] += base_action[0]
        self.plane[2] += base_action[0]
        self.plane[1] += base_action[1]
        self.plane[3] += base_action[1]
        #next_coords = self.canvas.coords(self.plane)  # next state
        next_coords = self.plane
        # reward function
        goal_coords = self.oval
        if next_coords == goal_coords:
            reward = 10 + self.euclidean(next_coords,goal_coords)
            done = True
        else:
            curr_x = int((next_coords[0] + next_coords[2]) // 2)
            curr_y = int((next_coords[1] + next_coords[3]) // 2)

            # Repeating values for padding
            if curr_x >= 512:
                curr_x = 511
            if curr_y >= 512:
                curr_y = 511

            table_lookup = self.maze_arr[curr_x,curr_y]
            
            if(table_lookup == 255):
                table_lookup = -1
            #table_lookup *= 255
            reward = table_lookup + self.euclidean(next_coords,goal_coords)
            done = False
       
        s_ = (np.array(next_coords[:2]) - np.array(self.oval[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def euclidean(self,coord,goal_coord):
        curr_x = (coord[0] + coord[2]) // 2
        curr_y = (coord[1] + coord[3]) // 2
        goal_x = (coord[0] + coord[2]) // 2
        goal_y = (coord[1] + coord[3]) // 2
        dist = ( (goal_x - curr_x)**2 + (goal_y - curr_y)**2 )**.5
        return 1/(dist + 1.0)