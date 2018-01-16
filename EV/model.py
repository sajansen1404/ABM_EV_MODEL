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
    def __init__(self, N = 50, width = 20, height = 20, n_poles = 10):
        self.num_agents = N
        self.grid = MultiGrid(width, height, False) #toroidal (for now)
        self.schedule = RandomActivation(self)
        self.vision = 10
        
        # Create Charge Pole agents
        for i in range(n_poles):
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            charge_pole = Charge_pole(i, (x,y), self)
            self.grid.place_agent(charge_pole, (x, y))
        
        # Create EV agents
        for i in range(self.num_agents):
            EV = EV_Agent(i, self, self.vision)
            self.schedule.add(EV)
            # Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(EV, (x, y))
        
            
        self.datacollector = DataCollector(
            agent_reporters={"Battery": lambda EV: EV.battery},
            model_reporters= {"Avg_Battery": mean_all_battery,
                                "lower25":lowest_25_percent,
                                "Num_agents": count_agents})
        #self.datacollector = DataCollector(data)

        self.running = True
            
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)





