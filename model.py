### model.py

import matplotlib.pyplot as plt
import numpy as np
import random
import math
from pyDOE import *
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from scipy.spatial import distance

from EV.agents import EV_Agent, Charge_pole
from EV.schedule import RandomActivationByBreed


# ouput the mean of all battery
def mean_all_battery(model):

    agent_battery_levels = [agent.battery for agent in model.schedule.agents]
    #x = sorted(agent_wealths)
    
    B = np.mean(agent_battery_levels)
    return B

def lowest_25_percent(model):
    agent_battery_levels = [agent.battery for agent in model.schedule.agents]
    #x = sorted(agent_wealths)
    return  np.percentile(agent_battery_levels, 25)

def specific_battery(model):
	for agent in model.schedule.agents:
		if agent.unique_id == 10:
			return agent.battery

def time_in_state(model):
    agent_time_in_state = [agent.time_in_state for agent in model.schedule.agents]
    return np.mean(agent_time_in_state)

def count_agents(model):  
    N = model.num_agents
    return N

  # gives back a list of n points in a circle of radius r
def PointsInCircum(r,n=100):
    return [(round(math.cos(2*np.pi/n*x)*r),round(math.sin(2*np.pi/n*x)*r)) for x in range(0,n+1)]

# Create the model
class EV_Model(Model):
    def __init__(self, N = 50, width = 20, height = 20, n_poles = 10, vision = 10, grid_positions = "random", initial_bravery = 10, battery_size = 25):
        self.num_agents = N
        self.grid = MultiGrid(width, height, False) #toroidal (for now)
        self.schedule = RandomActivationByBreed(self)
        self.vision = vision
        self.grid_size = width



        #grid_positions = "LHS"


        # circle_
        if grid_positions == "circle":

            center_grid = (int(width/2), int(height/2))
            circle_list = PointsInCircum(round(self.grid_size/4), n_poles)
            for i, coord in enumerate(circle_list):
                new_coord = (coord[0] + center_grid[0],coord[1] + center_grid[1])

                charge_pole = Charge_pole(i, new_coord, self)
                self.grid.place_agent(charge_pole, new_coord)
                
        elif grid_positions == "big circle":
            center_grid = (int(width/2), int(height/2))
            circle_list = PointsInCircum(round(self.grid_size/2-1), n_poles)
            for i, coord in enumerate(circle_list):
                new_coord = (coord[0] + center_grid[0],coord[1] + center_grid[1])

                charge_pole = Charge_pole(i, new_coord, self)
                self.grid.place_agent(charge_pole, new_coord)

        # Create Charge Pole agents
        elif grid_positions == "random":
            for i in range(n_poles):
                # Add the agent to a random grid cell
                empty_coord = self.grid.find_empty()
                charge_pole = Charge_pole(i,empty_coord, self)
                self.grid.place_agent(charge_pole, empty_coord)

        elif grid_positions == "LHS":
            print(n_poles)
            coord_list =  np.round(lhs(2, samples = n_poles, criterion = "m")*(self.grid_size-1))
            print(len(coord_list))
            for i in range(n_poles):
                coord = tuple((int(coord_list[i][0]), int(coord_list[i][1])))
                if self.grid.is_cell_empty(coord):
                    charge_pole = Charge_pole(i,coord, self)
                    self.grid.place_agent(charge_pole, coord)
                else:
                    empty_coord = self.grid.find_empty()
                    charge_pole = Charge_pole(i,empty_coord, self)
                    self.grid.place_agent(charge_pole, empty_coord)




            

        # Create EV agents
        for i in range(self.num_agents):
            
            # Add the agent to a random grid cell
            empty_coord = self.grid.find_empty()
            home_pos = self.grid.find_empty()
            work_pos = self.grid.find_empty()

            EV = EV_Agent(i, self, self.vision, np.array(home_pos), np.array(work_pos), initial_bravery, battery_size)
            self.schedule.add(EV)
            
            self.grid.place_agent(EV, home_pos)


        self.datacollector = DataCollector(
            agent_reporters={"Battery": lambda EV: EV.battery},
            model_reporters= {"Avg_Battery": mean_all_battery,

                                "lower25": lowest_25_percent,
                                "timeInState": time_in_state,
                                "unique_battery":specific_battery,

                                "Num_agents": count_agents,
                                "EVs": lambda m: m.schedule.get_breed_count(EV_Agent)})
        #self.datacollector = DataCollector(data)

        self.running = True
            
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def get_distance(self, pos_1, pos_2):
        """ Get the distance between two point, accounting for toroidal space.

        Args:
            pos_1, pos_2: Coordinate tuples for both points.

        """
        pos_1 = np.array(pos_1)
        pos_2 = np.array(pos_2)
        if self.grid.torus:
            pos_1 = (pos_1 - int(self.grid.width/2)) % self.grid.width
            pos_2 = (pos_2 - int(self.grid.height/2)) % self.grid.height
        return np.linalg.norm(pos_1 - pos_2)





