### model.py

import matplotlib.pyplot as plt
import numpy as np
import random
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

def count_agents(model):  
    N = model.num_agents
    return N


# Create the model
class EV_Model(Model):
    def __init__(self, N = 50, width = 20, height = 20, n_poles = 10, vision = 10):
        self.num_agents = N
        self.grid = MultiGrid(width, height, False) #toroidal (for now)
        self.schedule = RandomActivationByBreed(self)
        self.vision = vision


        # All grid points: (Terribly inefficient, but works.)
        all_grid_points = []
        for i in np.arange(self.grid.width):
            for j in np.arange(self.grid.height):
                all_grid_points.append([i, j])
        np.random.shuffle(all_grid_points)


        # Create Charge Pole agents
        index = 0
        for i in range(n_poles):
            x, y = all_grid_points[index]
            index = index + 1

            charge_pole = Charge_pole(i, (x,y), self)
            self.grid.place_agent(charge_pole, (x, y))

        # Create list of all possible homes and workplaces, and choose two randomly per agent

        # Create EV agents
        for i in range(self.num_agents):
            home_pos = all_grid_points[index]
            index = index + 1
            work_pos = all_grid_points[index]
            index = index + 1

            EV = EV_Agent(i, self, self.vision, home_pos, work_pos)
            self.schedule.add(EV)
            # Add the agent to a random grid cell
            self.grid.place_agent(EV, home_pos)

        self.datacollector = DataCollector(
            agent_reporters={"Battery": lambda EV: EV.battery},
            model_reporters= {"Avg_Battery": mean_all_battery,
                                "lower25":lowest_25_percent,
                                "Num_agents": count_agents,
                                "EVs": lambda m: m.schedule.get_breed_count(EV_Agent)})
        #self.datacollector = DataCollector(data)

        self.running = True
            
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)





