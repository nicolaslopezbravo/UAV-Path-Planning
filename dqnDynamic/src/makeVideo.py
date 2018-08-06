import cv2
import numpy as np
from PIL import Image
import random
import os
CAR_VALUE = -500
ROAD_VALUE = 255
cars = []
UNIT = 1   # pixels
MAZE_H = 300  # grid height
MAZE_W = 300  # grid width
MAX_VAL= 1000
MIN_VAL=-10000
ROAD_REWARD = -1
CAR_REWARD = -2
BUILDING_REWARD = 1

class Car():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.velocity = random.randint(1,3)
        direction = random.randint(1,2)
        self.status = 1
        if direction % 2 == 0:
            self.velocity*= -1

    def update_map(self,map_file):
        if self.status == 0:
            return map_file

        x_ = self.x
        y_ = self.y
        # eliminate the car
        map_file [self.x,self.y] = ROAD_VALUE
        x_ +=self.velocity

        # check for boundaries
        if x_ >= 300 or x_ < 0 or y_ < 0 or y_ >= 300:
            self.status = 0
        else:
            # move to next available spot
            if map_file[x_,self.y] == ROAD_VALUE:
                map_file[x_,self.y] = CAR_VALUE
            else:
                y_ += self.velocity
                if map_file[self.x,y_] == ROAD_VALUE:
                    map_file[self.x,y_] = CAR_VALUE
                elif map_file[x_,y_] == ROAD_VALUE:
                    map_file[x_,y_] = CAR_VALUE
                else:
                    self.status = 0
                
            self.x = x_
            self.y = y_

        return map_file
        # else we just assume the car left the map         


class DynamicMap(object):
    def __init__(self):
         # make dynamic map
        self.map_file = np.loadtxt('map.txt',dtype=int)
        # get road locations
        self.road_locations = []
        self.get_roads()
        # put cars at random 
        self.cars = []  #100 cars of size pixel = 1
        self.generate_cars(100,1)
        self.run_map(300)
        self.time = 0

    def run_map(self,epochs):
        path = "DQN_path_imgs/"
        if not os.path.isdir(path):
            os.mkdir(path)
        
        img = np.zeros_like(cv2.imread('test.jpg'))

        for episode in range(epochs):
            print("starting epoch ", episode)
            image_file = img.copy()
            for i in range(300):
                for j in range(300):
                    if self.map_file[i,j]== 0:
                        image_file[i,j] = [0,0,0] #nothing
                    elif self.map_file[i,j] == 1:
                        image_file[i,j] = [0,255,0] #green
                    elif self.map_file[i,j] == 255:
                        image_file[i,j] = [0,0,255] #red
                    elif self.map_file[i,j] == CAR_VALUE:
                        image_file[i,j] = [255,255,255] #black
            
            cv2.imwrite(path + str(episode)+".jpg",image_file)
            #cvw.imshow("t",image_file)
            print(len(self.cars))
            for car in self.cars:
                if car.status != 0:
                    self.map_file = car.update_map(self.map_file)
                else:
                    self.cars.remove(car)
    
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
        print(len(self.cars))

dyn = DynamicMap()