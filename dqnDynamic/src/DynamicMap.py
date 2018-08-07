import cv2
import numpy as np
from PIL import Image
import random
import os
CAR_VALUE = -500
ROAD_VALUE = 255
UNIT = 1   # pixels
MAZE_H = 300  # grid height
MAZE_W = 300  # grid width
MAX_VAL= 1000
MIN_VAL=-10000
ROAD_REWARD = -1
CAR_REWARD = -2
BUILDING_REWARD = 1
# class car for synthetically generated data, however real data exists in the path planning folder of the matlab code
class Car():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.velocity = random.randint(1,3)
        direction = random.randint(1,2)
        self.status = 1
        if direction % 2 == 0:
            self.velocity*= -1

    def update_map(self, map_file):
        if self.status == 0:
            return map_file

        x_ = self.x
        y_ = self.y
        # eliminate the car
        map_file [self.x,self.y] = ROAD_VALUE
        x_ +=self.velocity
        
        # do something about out of bounds
        if x_ >= 300 or x_ < 0 or y_ < 0 or y_ >= 300 or self.x >= 300 or self.y >= 300 or self.x < 0 or self.y < 0:
            self.status = 0
        else:
            # move to next available spot
            if map_file[x_,self.y] == ROAD_VALUE:
                map_file[x_,self.y] = CAR_VALUE
            else:
                y_ += self.velocity
                if y_ < 300:
                    if map_file[self.x,y_] == ROAD_VALUE:
                        map_file[self.x,y_] = CAR_VALUE
                    elif map_file[x_,y_] == ROAD_VALUE:
                        map_file[x_,y_] = CAR_VALUE
                    self.y = y_
                else:
                    self.status = 0
            
            self.x = x_

        return map_file
        # else we just assume the car left the map         

class DynamicMap(object):
    def __init__(self,origin,destination,iteration):
        # make dynamic map
        self.map_file = np.loadtxt('map.txt',dtype=int)
        # get road locations
        self.road_locations = []
        self.get_roads()
        # put cars at random 
        self.cars = []  #100 cars of size pixel = 1
        self.generate_cars(100,1)

        # normal map operations
        self.action_space = ['u', 'd', 'l', 'r','ur','rd','ld','ul'] #accounting for diagonals
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.origin = np.array(origin) # starting position [118, 85]
        self.destination = destination
        self.benchmark_distance = ((self.destination[0] - self.origin[0])**2 + (self.destination[1] - self.origin[1])**2 )**.5
        self.map_file[self.destination[0],self.destination[1]] = 1000
        self.plane = [self.origin[0], self.origin[1]]
        self.iteration = str(iteration)
        self.time_step = 0
       
    def reset(self,episode):
        self.plane = [self.origin[0], self.origin[1]]
        self.episode = episode # episode means epoch here
        observation = (np.array(self.plane) - np.array(self.destination))/(MAZE_H*UNIT) # observation is the distance from the target 
        
        self.map_file = np.loadtxt('map.txt',dtype=int)
        self.road_locations = []
        self.get_roads()
        # put cars at random 
        self.cars = []  #100 cars of size pixel = 1
        self.generate_cars(100,1)
        #self.write_imgs()

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
       
        #self.write_imgs()
        # reward function
        goal_coords = self.destination
        if next_coords == goal_coords:
            reward = MAX_VAL            
            done = True
        else:
            curr_x = next_coords[0]
            curr_y = next_coords[1]

            table_lookup = self.map_file[curr_x,curr_y] # read the image rewards from the combined image
            
            if(table_lookup == 255):
                table_lookup = ROAD_REWARD  # for roads
            if(table_lookup == 1):
                table_lookup = BUILDING_REWARD  # for buildings
            if(table_lookup == CAR_VALUE):
                table_lookup = CAR_REWARD
               
            dist = self.euclidean(next_coords,goal_coords)
            if(((-1)*dist) < self.benchmark_distance):
                dist = 2.77**(dist)
            # calculating the reward function
            reward = table_lookup + (dist/(MAZE_H*UNIT)) # do some cases with h*w
            done = False

            if table_lookup == MIN_VAL or table_lookup == CAR_VALUE:
                done = True
        s_ = (np.array(next_coords) - np.array(self.destination))/(MAZE_H*UNIT)
        print(s_)
        self.update_all_cars()
        self.time_step += 1        
        return s_, reward, done

    def update_all_cars(self):
        for car in self.cars:
            if car.status != 0:
                self.map_file = car.update_map(self.map_file)
            else:
                self.cars.remove(car)

    def write_imgs(self):
        map_file = np.loadtxt('map.txt',dtype=int)
        image_file = np.zeros_like(cv2.imread('test.jpg'))

        for i in range(300):
            for j in range(300):
                if map_file[i,j]== 0:
                    image_file[i,j] = [0,0,0] #nothing
                elif map_file[i,j] == 1:
                    image_file[i,j] = [0,255,0] #green
                elif map_file[i,j] == 255:
                    image_file[i,j] = [0,0,255] #red
                elif map_file[i,j] == CAR_VALUE:
                    image_file[i,j] = [255,255,255] #white
        path = "DQN_path_imgs/"+self.iteration + "/"
        if not os.path.isdir(path):
            os.mkdir(path)

        x = self.plane[0]
        y = self.plane[1]


        # here is where we can check objects encountered
        cv2.circle(image_file,(x,y),5,(255,0,255),-1)
        cv2.imwrite(path + str(self.time_step)+".jpg",image_file)
    
    def euclidean(self,coord,goal_coord):
        curr_x = coord[0]
        curr_y = coord[1]
        goal_x = goal_coord[0]
        goal_y = goal_coord[1]
        dist = ( (goal_x - curr_x)**2 + (goal_y - curr_y)**2 )**.5
        return -dist

    def get_roads(self):
        for i in range(300):
            for j in range(300):
                if self.map_file[i,j]== 255:
                    self.road_locations.append((i,j))
                    

    def generate_cars(self,num_cars,size):
        length = len(self.road_locations)
        occupied_space = []
        for i in range(num_cars):
            index = random.randint(1,length)
            while index in occupied_space:
                index = random.randint(1,length)
            self.put_car(index)
            occupied_space.append(index)

    def put_car(self,index):
        pt = self.road_locations[index]
        x = pt[0]
        y = pt[1]
        self.map_file[x,y] = CAR_VALUE
        car = Car(x,y)
        self.cars.append(car)

