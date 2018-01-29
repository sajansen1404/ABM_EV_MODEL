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

def time_in_state(model):
    agent_time_in_state = [agent.time_in_state for agent in model.schedule.agents]
    return np.mean(agent_time_in_state)

def count_agents(model):  
    N = model.num_agents
    return N


# Create the model
class EV_Model(Model):
    def __init__(self, N = 50, width = 20, height = 20, n_poles = 10, vision = 10):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True) #toroidal (for now)
        self.schedule = RandomActivationByBreed(self)
        self.vision = vision

        # Create Charge Pole agents
        for i in range(n_poles):
            # Add the agent to a random grid cell
            empty_coord = self.grid.find_empty()
            charge_pole = Charge_pole(i,empty_coord, self)
            self.grid.place_agent(charge_pole, empty_coord)
        
        # Create EV agents
        for i in range(self.num_agents):
            
            # Add the agent to a random grid cell
            empty_coord = self.grid.find_empty()
            home_pos = self.grid.find_empty()
            work_pos = self.grid.find_empty()
            EV = EV_Agent(i, self, self.vision, np.array(home_pos), np.array(work_pos))
            self.schedule.add(EV)
            
            self.grid.place_agent(EV, empty_coord)

        self.datacollector = DataCollector(
            agent_reporters={"Battery": lambda EV: EV.battery},
            model_reporters= {"Avg_Battery": mean_all_battery,
                                "lower25": lowest_25_percent,
                                "timeInState": time_in_state,
                                "Num_agents": count_agents,
                                "EVs": lambda m: m.schedule.get_breed_count(EV_Agent)})
        #self.datacollector = DataCollector(data)

        self.running = True
            
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)





